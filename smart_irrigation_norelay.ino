// Smart irrigation, relay-free build, with over-watering protection.
// The pump decision is shown on the built-in LED, printed to Serial, sent to ThingsBoard,
// Telegram and the laptop dashboard. Nothing switches the pump: you do it by hand.
//
// Over-watering protection (all tunable below):
//   - start/stop band: advise ON below SOIL_START_PCT, keep ON until SOIL_STOP_PCT
//   - early stop if soil is rising so fast it will overshoot the target (probe lag)
//   - max session length, cooldown between sessions, daily pump-time cap
//   - rain safety rule looks at the highest chance over the next 12 hours

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <WiFiClientSecure.h>
#include <time.h>
#define TG_BOT_TOKEN "8995148032:AAH3oqVVIZ_dZ_UacCER0xbX_Eh6cPLtH5A"
#define TG_CHAT_ID "8284186097"

#if defined(TG_BOT_TOKEN) && defined(TG_CHAT_ID)
  #define TELEGRAM_ENABLED 1
#else
  #define TELEGRAM_ENABLED 0
#endif

// ---------------- build options ----------------
#define PUMP_HARDWARE 0          // 0 = no pump switch (LED + log only), 1 = drive GPIO23
#define PUMP_ACTIVE_LOW 0        // only used when PUMP_HARDWARE is 1
#define LDR_MODE 0               // 0 = no LDR, 1 = LDR + 10k on GPIO35, 2 = LDR only on GPIO32 (internal pull-up)
#define TEST_FAKE_RESERVOIR_PCT -1  // 0..100 fakes the tank level; -1 = use the sensor
#define DEMO_FAST_CYCLE 1        // 1 = cycle every 15 s and short timers, 0 = every 5 min and real timers
#define TEST_FORCE_ML_RESULT -1  // CHANGE THIS: -1 = use the real model
#define TZ_OFFSET_S 19800        // India (UTC+5:30), used to reset the daily pump-time counter at midnight

#define WIFI_SSID "realme 14 Pro lite 5G"
#define WIFI_PASSWORD "iambeautiful"

#define OWM_API_KEY "0b421b2fa8a4e8e9f016f56df19ffe5a"
#define OWM_LAT "12.97"
#define OWM_LON "79.15"
#define TB_SERVER "https://eu.thingsboard.cloud"
#define TB_TOKEN "tb_fMEYZkeh2KGXHnLxF51XJ_tPrUU5IOLRxYT4wZWo5vlzj3IPLaEC7YQtXROFcGgU7CGlAxndtRqf_wpgF3PJxg"
#define ML_SERVER_URL "http://10.233.212.56:8000/predict"
#define REPORT_URL "http://10.233.212.56:8000/report"

// ---------------- pins ----------------
#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_PIN 36
#define TRIG_PIN 18
#define ECHO_PIN 19
#define RELAY_PIN 23
#define STATUS_LED 2             // built-in LED on most ESP32 Dev boards

#if LDR_MODE == 1
  #define LDR_PIN 35
#elif LDR_MODE == 2
  #define LDR_PIN 32
#endif

// ---------------- thresholds ----------------
const float RESERVOIR_HEIGHT_CM = 30.0;
const float SAFETY_RESERVOIR_PCT = 10.0;
const float RAIN_FORECAST_THRESHOLD = 70.0;   // checked against the highest chance in the next 12 h
const float SOIL_START_PCT = 35.0;            // advise ON only below this
const float SOIL_STOP_PCT = 55.0;             // keep ON until soil reaches this
const int LDR_PEAK_SUN_THRESHOLD = 3000;

DHT dht(DHTPIN, DHTTYPE);

unsigned long lastCycle = 0;
#if DEMO_FAST_CYCLE
const unsigned long CYCLE_INTERVAL  = 15000;
const unsigned long MAX_SESSION_MS  = 2UL * 60 * 1000;        // 2 min
const unsigned long COOLDOWN_MS     = 1UL * 60 * 1000;        // 1 min
const unsigned long MAX_DAILY_ON_MS = 10UL * 60 * 1000;       // 10 min
#else
const unsigned long CYCLE_INTERVAL  = 300000;
const unsigned long MAX_SESSION_MS  = 20UL * 60 * 1000;       // 20 min
const unsigned long COOLDOWN_MS     = 2UL * 60 * 60 * 1000;   // 2 h for water to soak in
const unsigned long MAX_DAILY_ON_MS = 60UL * 60 * 1000;       // 60 min per day
#endif

struct Decision {
  bool pumpOn;
  const char* reason;
};

// ---------------- state ----------------
int lastMl = -1;                 // -1 = the model was not asked this cycle
float rainMax12 = 0;             // highest rain chance over the next 12 h
bool adviceOn = false;           // are we currently advising "pump ON"
unsigned long sessionStartMs = 0;
bool cooling = false;
unsigned long coolStartMs = 0;
unsigned long dayOnMs = 0;       // pump-on time finished today
int lastDay = -1;
float lastSoil = -1;
bool lastPumpOn = false;
bool lastSafetyAlert = false;

void setPump(bool on) {
  digitalWrite(STATUS_LED, on ? HIGH : LOW);
#if PUMP_HARDWARE
  bool level = PUMP_ACTIVE_LOW ? !on : on;
  digitalWrite(RELAY_PIN, level ? HIGH : LOW);
#else
  Serial.println(on ? ">>> PUMP SHOULD BE ON: switch the pump on by hand <<<"
                    : ">>> pump should be off <<<");
#endif
}
void sendTelegram(const String& text);

void setup() {
  Serial.begin(115200);
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);
#if PUMP_HARDWARE
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, PUMP_ACTIVE_LOW ? HIGH : LOW);   // pump off at boot
#endif
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
#if LDR_MODE == 2
  pinMode(LDR_PIN, INPUT_PULLUP);
#endif
  dht.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  configTime(TZ_OFFSET_S, 0, "pool.ntp.org", "time.google.com");   // clock for the daily counter
  sendTelegram("ESP32 irrigation monitor is online.");
}

float readSoilMoisturePct() {
  int raw = analogRead(SOIL_PIN);
  return constrain(map(raw, 4095, 1200, 0, 100), 0, 100);
}

// Brighter = higher number. With no LDR this returns 0, so the peak-sun hold never triggers.
int readSolarIntensity() {
#if LDR_MODE == 1
  return analogRead(LDR_PIN);
#elif LDR_MODE == 2
  return 4095 - analogRead(LDR_PIN);
#else
  return 0;
#endif
}

// Returns 0..100, or -1 if the sensor gives no echo.
float readReservoirLevelPct() {
  if (TEST_FAKE_RESERVOIR_PCT >= 0) return TEST_FAKE_RESERVOIR_PCT;
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  float distanceCm = duration * 0.0343 / 2.0;
  float waterHeight = RESERVOIR_HEIGHT_CM - distanceCm;
  float pct = (waterHeight / RESERVOIR_HEIGHT_CM) * 100.0;
  return constrain(pct, 0, 100);
}

// Returns the rain chance of the NEXT 3-hour block (what the model was trained on) and
// sets rainMax12 to the highest chance over the next 12 hours (used by the safety rule).
float fetchRainForecastPct() {
  rainMax12 = 0;
  if (WiFi.status() != WL_CONNECTED) return 0;
  HTTPClient http;
  String url = String("http://api.openweathermap.org/data/2.5/forecast?lat=") +
               OWM_LAT + "&lon=" + OWM_LON + "&cnt=4&appid=" + OWM_API_KEY + "&units=metric";
  http.begin(url);
  int code = http.GET();
  float firstBlock = 0;
  if (code == 200) {
    StaticJsonDocument<64> filter;
    filter["list"][0]["pop"] = true;
    DynamicJsonDocument doc(1024);
    DeserializationError err = deserializeJson(doc, http.getString(), DeserializationOption::Filter(filter));
    if (!err) {
      bool first = true;
      for (JsonObject item : doc["list"].as<JsonArray>()) {
        float p = (item["pop"] | 0.0f) * 100.0f;
        if (first) { firstBlock = p; first = false; }
        if (p > rainMax12) rainMax12 = p;
      }
    } else {
      Serial.println("OWM parse error");
    }
  } else {
    Serial.printf("OWM request failed, code=%d\n", code);
  }
  http.end();
  return firstBlock;
}

// Calls the PC-hosted FastAPI /predict endpoint. Returns 0 or 1.
// On any failure it fails SAFE by returning 0 (do not irrigate).
int mlPredict(float soil, float temp, float hum, int light, float reservoir, float rain) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("ML predict skipped: WiFi not connected, defaulting to 0");
    return 0;
  }

  HTTPClient http;
  String url = String(ML_SERVER_URL) +
               "?soil=" + String(soil, 2) +
               "&temp=" + String(temp, 2) +
               "&hum=" + String(hum, 2) +
               "&light=" + String(light) +
               "&reservoir=" + String(reservoir, 2) +
               "&rain=" + String(rain, 2);

  http.begin(url);
  http.setTimeout(5000);
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

// ---------------- watering session bookkeeping ----------------
void rollDay() {
  struct tm t;
  int day = getLocalTime(&t, 0) ? t.tm_yday : (int)(millis() / 86400000UL);
  if (lastDay != -1 && day != lastDay) dayOnMs = 0;   // new day, reset the daily pump-time counter
  lastDay = day;
}

unsigned long onTodayMs() {
  return dayOnMs + (adviceOn ? millis() - sessionStartMs : 0);
}

unsigned long coolLeftMs() {
  if (!cooling) return 0;
  unsigned long elapsed = millis() - coolStartMs;
  return elapsed < COOLDOWN_MS ? COOLDOWN_MS - elapsed : 0;
}

Decision turnOn(const char* reason) {
  if (!adviceOn) {
    adviceOn = true;
    sessionStartMs = millis();
  }
  return {true, reason};
}

Decision turnOff(const char* reason) {
  if (adviceOn) {
    adviceOn = false;
    dayOnMs += millis() - sessionStartMs;
  }
  return {false, reason};
}

// Same as turnOff, but a finished watering session also starts the cooldown.
Decision turnOffCool(const char* reason) {
  if (adviceOn) {
    cooling = true;
    coolStartMs = millis();
  }
  return turnOff(reason);
}

Decision decideIrrigation(float soil, float temp, float hum, int light, float reservoir, float rain) {
  lastMl = -1;
  rollDay();
  float rise = (lastSoil >= 0) ? soil - lastSoil : 0;   // how much soil moved since the last cycle
  lastSoil = soil;

  // hard safety interlocks: always checked, and they also stop a running session
  if (reservoir < 0) return turnOff("reservoir sensor error");
  if (reservoir < SAFETY_RESERVOIR_PCT) return turnOff("reservoir low");
  if (rainMax12 >= RAIN_FORECAST_THRESHOLD) return turnOff("rain forecast");
  if (LDR_MODE != 0 && light >= LDR_PEAK_SUN_THRESHOLD) return turnOff("peak sun hold");
  if (onTodayMs() >= MAX_DAILY_ON_MS) return turnOff("daily limit reached");

  // a session is running: keep going until a stop condition is met
  if (adviceOn) {
    if (soil >= SOIL_STOP_PCT) return turnOffCool("soil reached target");
    if (millis() - sessionStartMs >= MAX_SESSION_MS) return turnOffCool("session time limit");
    // the probe lags, so if soil is rising fast it will overshoot: stop early
    if (rise > 0 && soil + rise >= SOIL_STOP_PCT) return turnOffCool("soil rising fast");
    return turnOn("watering in progress");
  }

  // not running: decide whether to start
  if (coolLeftMs() > 0) return turnOff("cooling down");
  cooling = false;
  if (soil >= SOIL_START_PCT) return turnOff("soil moist enough");

  int mlOutput = (TEST_FORCE_ML_RESULT >= 0) ? TEST_FORCE_ML_RESULT
               : mlPredict(soil, temp, hum, light, reservoir, rain);
  lastMl = mlOutput;

  if (mlOutput == 1) return turnOn("soil dry and model says irrigate");
  return turnOff("model says no");
}

void sendToThingsBoard(float soil, float temp, float hum, int light, float reservoir, float rain, const Decision& d) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(TB_SERVER) + "/api/v1/" + TB_TOKEN + "/telemetry";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");

  DynamicJsonDocument doc(768);
  doc["soil_moisture"] = soil;
  doc["temperature"] = temp;
  doc["humidity"] = hum;
  doc["solar_intensity"] = light;
  doc["reservoir_level"] = reservoir;
  doc["rain_forecast"] = rain;
  doc["rain_forecast_12h"] = rainMax12;
  doc["pump_state"] = d.pumpOn ? 1 : 0;
  doc["pump_on_today_min"] = onTodayMs() / 60000.0;
  doc["decision_reason"] = d.reason;

  String payload;
  serializeJson(doc, payload);
  int code = http.POST(payload);
  if (code < 0) {
    Serial.printf("ThingsBoard POST failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

String urlEncode(const String& s) {
  String out;
  char buf[4];
  for (size_t i = 0; i < s.length(); i++) {
    char c = s[i];
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out += c;
    } else {
      snprintf(buf, sizeof(buf), "%%%02X", (uint8_t)c);
      out += buf;
    }
  }
  return out;
}

// Sends a Telegram message. Uses an unverified TLS connection (setInsecure),
// which is fine for alerts but does not check the server's certificate.
void sendTelegram(const String& text) {
#if TELEGRAM_ENABLED
  if (WiFi.status() != WL_CONNECTED) return;
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  String url = String("https://api.telegram.org/bot") + TG_BOT_TOKEN +
               "/sendMessage?chat_id=" + TG_CHAT_ID + "&text=" + urlEncode(text);
  if (http.begin(client, url)) {
    int code = http.GET();
    if (code != 200) Serial.printf("Telegram send failed, code=%d\n", code);
    http.end();
  }
#else
  Serial.println("(Telegram not configured, alert not sent)");
#endif
}

void sendAlerts(const Decision& d, float soil, float temp, float hum, float reservoir, float rain) {
  char line[160];
  snprintf(line, sizeof(line), "Soil %.0f%%, %.1f C, humidity %.0f%%, tank %.0f%%, rain chance %.0f%% (12 h max %.0f%%)",
           soil, temp, hum, reservoir, rain, rainMax12);

  if (d.pumpOn && !lastPumpOn) {
    sendTelegram(String("TURN THE PUMP ON. Soil is dry and the model says irrigate.\n") + line);
  } else if (!d.pumpOn && lastPumpOn) {
    String msg = String("TURN THE PUMP OFF. Reason: ") + d.reason;
    if (strcmp(d.reason, "session time limit") == 0) msg += ". Soil never reached the target: check the probe and the pipe.";
    if (strcmp(d.reason, "soil rising fast") == 0) msg += ". Stopping early so the soil does not overshoot.";
    sendTelegram(msg + "\n" + line);
  }
  lastPumpOn = d.pumpOn;

  bool safety = strcmp(d.reason, "reservoir low") == 0 || strcmp(d.reason, "reservoir sensor error") == 0;
  if (safety && !lastSafetyAlert) {
    sendTelegram(String("WARNING: ") + d.reason + ". Pump must stay off.\n" + line);
  }
  lastSafetyAlert = safety;
}

void sendReport(float soil, float temp, float hum, int light,
                float reservoir, float rain, const Decision& d) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(REPORT_URL);
  http.setTimeout(5000);
  http.addHeader("Content-Type", "application/json");
  DynamicJsonDocument doc(1024);
  doc["soil"] = soil;  doc["temp"] = temp;  doc["hum"] = hum;
  doc["light"] = light;
  doc["tank"] = reservoir;  doc["rain"] = rain;  doc["rain_max"] = rainMax12;
  doc["ml"] = lastMl;  doc["pump"] = d.pumpOn;  doc["reason"] = d.reason;
  doc["soil_start"] = SOIL_START_PCT;
  doc["soil_stop"] = SOIL_STOP_PCT;
  doc["rain_limit"] = RAIN_FORECAST_THRESHOLD;
  doc["tank_limit"] = SAFETY_RESERVOIR_PCT;
  doc["session_min"] = adviceOn ? (millis() - sessionStartMs) / 60000.0 : 0.0;
  doc["session_cap_min"] = MAX_SESSION_MS / 60000.0;
  doc["on_today_min"] = onTodayMs() / 60000.0;
  doc["daily_cap_min"] = MAX_DAILY_ON_MS / 60000.0;
  doc["cool_left_min"] = coolLeftMs() / 60000.0;
  doc["cycle_s"] = CYCLE_INTERVAL / 1000;
  String body;
  serializeJson(doc, body);
  int code = http.POST(body);
  if (code < 0) Serial.printf("Report POST failed: %s\n", http.errorToString(code).c_str());
  http.end();
}

void loop() {
  if (millis() - lastCycle < CYCLE_INTERVAL && lastCycle != 0) return;
  lastCycle = millis();

  float soil = readSoilMoisturePct();
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int light = readSolarIntensity();
  float reservoir = readReservoirLevelPct();
  float rain = fetchRainForecastPct();

  if (isnan(temp) || isnan(hum)) {
    Serial.println("DHT read failed, skipping this cycle");
    setPump(false);
    return;
  }

  Decision d = decideIrrigation(soil, temp, hum, light, reservoir, rain);
  setPump(d.pumpOn);
  sendAlerts(d, soil, temp, hum, reservoir, rain);
  sendToThingsBoard(soil, temp, hum, light, reservoir, rain, d);
  sendReport(soil, temp, hum, light, reservoir, rain, d);

  Serial.printf("Soil:%.1f Temp:%.1f Hum:%.1f Light:%d Reservoir:%.1f Rain:%.1f/%.1f Pump:%s (%s) today:%.1f min\n",
                soil, temp, hum, light, reservoir, rain, rainMax12, d.pumpOn ? "ON" : "OFF", d.reason,
                onTodayMs() / 60000.0);
}
