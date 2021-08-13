#ifdef ENABLE_DEBUG
       #define DEBUG_ESP_PORT Serial
       #define NODEBUG_WEBSOCKETS
       #define NDEBUG
#endif 

#include <WiFi.h>
#include "SinricPro.h"
#include "SinricProLight.h"

#include <map>

#define WIFI_SSID  ""
#define WIFI_PASS  ""
#define APP_KEY    "" // Get from Sinricpro dashboard
#define APP_SECRET "" // Get from Sinricpro dashboard

#define device_ID_1 "60cf25278cf8a303b93a142c"

#define RelayPin1 5

#define wifiLed 2

typedef struct {      // struct for the std::map below
  int relayPIN;
} deviceConfig_t;

std::map<String, deviceConfig_t> devices = {
    {device_ID_1, {  RelayPin1 }}   
};

void setupRelays() { 
  for (auto &device : devices) {           // for each device (relay)
    int relayPIN = device.second.relayPIN; // get the relay pin
    pinMode(relayPIN, OUTPUT);             // set relay pin to OUTPUT
    digitalWrite(relayPIN, HIGH);
  }
}

bool onPowerState(String deviceId, bool &state){
  Serial.printf("%s: %s\r\n", deviceId.c_str(), state ? "on" : "off");
  int relayPIN = devices[deviceId].relayPIN; // get the relay pin for corresponding device
  digitalWrite(relayPIN, !state);             // set the new relay state
  return true;
}

void setupWiFi()
{
  Serial.printf("\r\n[Wifi]: Connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.printf(".");
    delay(250);
  }
  digitalWrite(wifiLed, HIGH);
  Serial.printf("connected!\r\n[WiFi]: IP-Address is %s\r\n", WiFi.localIP().toString().c_str());
}

void setupSinricPro()
{
  for (auto &device : devices)
  {
    const char *deviceId = device.first.c_str();
    SinricProLight &bulb = SinricPro[deviceId];
    bulb.onPowerState(onPowerState);
  }

  SinricPro.begin(APP_KEY, APP_SECRET);
  SinricPro.restoreDeviceStates(true);
}

void setup()
{
  Serial.begin(9600);

  pinMode(wifiLed, OUTPUT);
  digitalWrite(wifiLed, LOW);

  setupRelays();
  setupWiFi();
  setupSinricPro();
}

void loop()
{
  SinricPro.handle();
}
