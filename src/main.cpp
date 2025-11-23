#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Adafruit PCA9685 driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo channels on the PCA9685
const uint8_t SERVO_CH[3] = {0, 2, 4};

// Servo pulse calibration (tune for your servos)
const uint16_t SERVO_MIN = 150; // ~0.9ms pulse
const uint16_t SERVO_MAX = 600; // ~2.4ms pulse

int currentAngles[3] = {0, 0, 0};
String inputBuffer = "";

// Convert angle (0..180) to PCA9685 pulse (0..4095)
uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setServoAngle(uint8_t servoIndex, int angle) {
  if (servoIndex >= 3) return;
  // Constrain angle to 0-65 degrees
  angle = constrain(angle, 0, 65);
  if (servoIndex == 2) {
    // The first effector needs a slight offset
    angle += 4;
  }

  if (servoIndex == 0) {
    angle += 3;
  }

  uint16_t pulse = angleToPulse(angle);
  pwm.setPWM(SERVO_CH[servoIndex], 0, pulse);
  currentAngles[servoIndex] = angle;
}

void parseAndApplyAngles(String data) {
  // Expected format: "90,90,90" or "45.5,90.2,60.1"
  int angles[3];
  int idx = 0;
  int startPos = 0;
  
  for (int i = 0; i < data.length() && idx < 3; i++) {
    if (data.charAt(i) == ',' || i == data.length() - 1) {
      String angleStr = (i == data.length() - 1) ? 
                        data.substring(startPos) : 
                        data.substring(startPos, i);
      angles[idx] = angleStr.toInt();
      idx++;
      startPos = i + 1;
    }
  }
  
  if (idx == 3) {
    for (int i = 0; i < 3; i++) {
      setServoAngle(i, angles[i]);
    }
    Serial.print(F("OK:"));
    Serial.print(currentAngles[0]); Serial.print(',');
    Serial.print(currentAngles[1]); Serial.print(',');
    Serial.println(currentAngles[2]);
  } else {
    Serial.println(F("ERR:Invalid format. Use: angle0,angle1,angle2"));
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  // Initialize servos to 0 degrees
  for (uint8_t i = 0; i < 3; ++i) {
    setServoAngle(i, 0);
    delay(50);
  }

  Serial.println(F("READY"));
  Serial.print(F("Channels:"));
  Serial.print(SERVO_CH[0]); Serial.print(',');
  Serial.print(SERVO_CH[1]); Serial.print(',');
  Serial.println(SERVO_CH[2]);
  Serial.println(F("Send: angle0,angle1,angle2"));
}

void loop() {
  // Read serial input
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        parseAndApplyAngles(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
  
  delay(5);
}
