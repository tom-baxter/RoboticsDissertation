#include <Wire.h>
#include <JY901.h>
#include <AltSoftSerial.h>
#include <NeoSWSerial.h>

// First sensor using AltSoftSerial (pins 8 = RX, 9 = TX)
AltSoftSerial sensor1Serial;
CJY901 sensor1;

// Second sensor using NeoSWSerial
#define SENSOR2_RX 4
#define SENSOR2_TX 5
NeoSWSerial sensor2Serial(SENSOR2_RX, SENSOR2_TX);
CJY901 sensor2;

float sensorValues[2][3] = {
  { 0, 0, 0 },
  { 0, 0, 0 }
};

unsigned long startTime = millis();
unsigned long nextUpdate = millis();
unsigned long currentTime = 0;

void setup() {
  Serial.begin(9600);
  sensor1Serial.begin(9600);
  sensor2Serial.begin(9600);
}

void loop() {

  // --- Read Sensor 1 ---
  while (sensor1Serial.available()) {
    sensor1.CopeSerialData(sensor1Serial.read());
  }

  // --- Read Sensor 2 ---
  while (sensor2Serial.available()) {
    sensor2.CopeSerialData(sensor2Serial.read());
  }

  currentTime = millis();
  if (currentTime > nextUpdate) {
    nextUpdate = currentTime + 50;

    // Store sensor 1
    sensorValues[0][0] = (float)sensor1.stcAngle.Angle[0] / 32768.0 * 180.0;
    sensorValues[0][1] = (float)sensor1.stcAngle.Angle[1] / 32768.0 * 180.0;
    sensorValues[0][2] = (float)sensor1.stcAngle.Angle[2] / 32768.0 * 180.0;

    // Store sensor 2
    sensorValues[1][0] = (float)sensor2.stcAngle.Angle[0] / 32768.0 * 180.0;
    sensorValues[1][1] = (float)sensor2.stcAngle.Angle[1] / 32768.0 * 180.0;
    sensorValues[1][2] = (float)sensor2.stcAngle.Angle[2] / 32768.0 * 180.0;

    // Print all values in a CSV line
    Serial.print(sensorValues[0][0], 3);
    Serial.print(",");
    Serial.print(sensorValues[0][1], 3);
    Serial.print(",");
    Serial.print(sensorValues[0][2], 3);
    Serial.print(",");
    Serial.print(sensorValues[1][0], 3);
    Serial.print(",");
    Serial.print(sensorValues[1][1], 3);
    Serial.print(",");
    Serial.println(sensorValues[1][2], 3);
  }
}
