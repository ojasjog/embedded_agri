#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <secrets.h>

#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_PIN 36
#define LEAF_PIN 34
#define LDR_PIN 35
#define TRIG_PIN 18
#define ECHO_PIN 19
#define RELAY_PIN 23

const char* WIFI_SSID = WIFI_SSID;
const char* WIFI_PASSWORD = WIFI_PASSWORD;

const char* OWM_API_KEY = OWM_API_KEY;
const char* OWM_LAT = "12.97";
const char* OWM_LON = "79.15";

const char* TB_SERVER = TB_SERVER;
const char* TB_TOKEN = TB_TOKEN;

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
  }
  http.end();
  return rainProb;
}

int xgboostPredict(float soil, float temp, float hum, float leaf, int light, float reservoir, float rain) {
  if (soil < SOIL_DRY_THRESHOLD && leaf < 50 && rain < RAIN_FORECAST_THRESHOLD) {
    return 1;
  }
  return 0;
}

bool decideIrrigation(float soil, float temp, float hum, float leaf, int light, float reservoir, float rain) {
  if (reservoir < SAFETY_RESERVOIR_PCT) return false;
  if (rain >= RAIN_FORECAST_THRESHOLD) return false;
  if (light >= LDR_PEAK_SUN_THRESHOLD) return false;
  int mlOutput = xgboostPredict(soil, temp, hum, leaf, light, reservoir, rain);
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
  http.POST(payload);
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

  bool pumpOn = decideIrrigation(soil, temp, hum, leaf, light, reservoir, rain);
  digitalWrite(RELAY_PIN, pumpOn ? HIGH : LOW);

  sendToThingsBoard(soil, temp, hum, leaf, light, reservoir, rain, pumpOn);

  Serial.printf("Soil:%.1f Temp:%.1f Hum:%.1f Leaf:%.1f Light:%d Reservoir:%.1f Rain:%.1f Pump:%s\n",
                soil, temp, hum, leaf, light, reservoir, rain, pumpOn ? "ON" : "OFF");
}
