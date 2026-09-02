#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include "secrets.h"   // WIFI_SSID, WIFI_PASSWORD, OWM_API_KEY, OWM_LAT, OWM_LON,
                        // TB_SERVER, TB_TOKEN, ML_SERVER_URL  (see secrets.h.example)

#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_PIN 36
#define LEAF_PIN 34
#define LDR_PIN 35
#define TRIG_PIN 18
#define ECHO_PIN 19
#define RELAY_PIN 23

const float RESERVOIR_HEIGHT_CM = 30.0;
const float SAFETY_RESERVOIR_PCT = 10.0;
const float RAIN_FORECAST_THRESHOLD = 70.0;
const float SOIL_DRY_THRESHOLD = 45.0;
const int LDR_PEAK_SUN_THRESHOLD = 3000;

DHT dht(DHTPIN, DHTTYPE);

unsigned long lastCycle = 0;
const unsigned long CYCLE_INTERVAL = 300000;

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  dht.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

float readSoilMoisturePct() {
  int raw = analogRead(SOIL_PIN);
  return constrain(map(raw, 4095, 1200, 0, 100), 0, 100);
}

float readLeafWetnessPct() {
  int raw = analogRead(LEAF_PIN);
  return constrain(map(raw, 4095, 0, 0, 100), 0, 100);
}

int readSolarIntensity() {
  return analogRead(LDR_PIN);
}

float readReservoirLevelPct() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  float distanceCm = duration * 0.0343 / 2.0;
  float waterHeight = RESERVOIR_HEIGHT_CM - distanceCm;
  float pct = (waterHeight / RESERVOIR_HEIGHT_CM) * 100.0;
  return constrain(pct, 0, 100);
}

float fetchRainForecastPct() {
  if (WiFi.status() != WL_CONNECTED) return 0;
  HTTPClient http;
  String url = String("http://api.openweathermap.org/data/2.5/forecast?lat=") +
               OWM_LAT + "&lon=" + OWM_LON + "&appid=" + OWM_API_KEY + "&units=metric";
  http.begin(url);
  int code = http.GET();
  float rainProb = 0;
  if (code == 200) {
    String payload = http.getString();
    DynamicJsonDocument doc(4096);
    deserializeJson(doc, payload);
    if (doc["list"][0].containsKey("pop")) {
      rainProb = doc["list"][0]["pop"].as<float>() * 100.0;
    }
  } else {
    Serial.printf("OWM request failed, code=%d\n", code);
  }
  http.end();
  return rainProb;
}

// Calls the PC-hosted FastAPI /predict endpoint instead of running the model
// on-device. Returns 0 or 1. On any failure (WiFi down, server unreachable,
// unexpected response) it fails SAFE by returning 0 (do not irrigate) --
// you don't want a network hiccup to accidentally leave the pump logic guessing.
int mlPredict(float soil, float temp, float hum, float leaf, int light, float reservoir, float rain) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("ML predict skipped: WiFi not connected, defaulting to 0");
    return 0;
  }

  HTTPClient http;
  String url = String(ML_SERVER_URL) +
               "?soil=" + String(soil, 2) +
               "&temp=" + String(temp, 2) +
               "&hum=" + String(hum, 2) +
               "&leaf=" + String(leaf, 2) +
               "&light=" + String(light) +
               "&reservoir=" + String(reservoir, 2) +
               "&rain=" + String(rain, 2);

  http.begin(url);
  http.setTimeout(5000);  // don't let a slow/dead server stall the whole cycle
  int code = http.GET();

  int result = 0;
  if (code == 200) {
    String body = http.getString();
    body.trim();
    if (body == "1") result = 1;
    else if (body == "0") result = 0;
    else {
      Serial.printf("ML server returned unexpected body: '%s', defaulting to 0\n", body.c_str());
    }
  } else {
    Serial.printf("ML server request failed, code=%d, defaulting to 0\n", code);
  }
  http.end();
  return result;
}

bool decideIrrigation(float soil, float temp, float hum, float leaf, int light, float reservoir, float rain) {
  // hard safety interlocks -- these are checked in firmware regardless of what
  // the model says, so a bad prediction can never override them
  if (reservoir < SAFETY_RESERVOIR_PCT) return false;
  if (rain >= RAIN_FORECAST_THRESHOLD) return false;
  if (light >= LDR_PEAK_SUN_THRESHOLD) return false;

  int mlOutput = mlPredict(soil, temp, hum, leaf, light, reservoir, rain);

  if (soil < SOIL_DRY_THRESHOLD && mlOutput == 1) return true;
  return false;
}

void sendToThingsBoard(float soil, float temp, float hum, float leaf, int light, float reservoir, float rain, bool pumpOn) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(TB_SERVER) + "/api/v1/" + TB_TOKEN + "/telemetry";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");

  DynamicJsonDocument doc(512);
  doc["soil_moisture"] = soil;
  doc["temperature"] = temp;
  doc["humidity"] = hum;
  doc["leaf_wetness"] = leaf;
  doc["solar_intensity"] = light;
  doc["reservoir_level"] = reservoir;
  doc["rain_forecast"] = rain;
  doc["pump_state"] = pumpOn ? 1 : 0;

  String payload;
  serializeJson(doc, payload);
  int code = http.POST(payload);
  if (code < 0) {
    Serial.printf("ThingsBoard POST failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

void loop() {
  if (millis() - lastCycle < CYCLE_INTERVAL && lastCycle != 0) return;
  lastCycle = millis();

  float soil = readSoilMoisturePct();
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  float leaf = readLeafWetnessPct();
  int light = readSolarIntensity();
  float reservoir = readReservoirLevelPct();
  float rain = fetchRainForecastPct();

  if (isnan(temp) || isnan(hum)) {
    Serial.println("DHT read failed, skipping this cycle");
    return;
  }

  bool pumpOn = decideIrrigation(soil, temp, hum, leaf, light, reservoir, rain);
  digitalWrite(RELAY_PIN, pumpOn ? HIGH : LOW);

  sendToThingsBoard(soil, temp, hum, leaf, light, reservoir, rain, pumpOn);

  Serial.printf("Soil:%.1f Temp:%.1f Hum:%.1f Leaf:%.1f Light:%d Reservoir:%.1f Rain:%.1f Pump:%s\n",
                soil, temp, hum, leaf, light, reservoir, rain, pumpOn ? "ON" : "OFF");
}
