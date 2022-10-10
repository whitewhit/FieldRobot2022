//#include "Wire.h"
//
//#define SLAVE_ADDRESS 0x04 

const int LMotor_PWM = 2;
const int LMotor_DIG = 3;
const int RMotor_PWM = 4;
const int RMotor_DIG = 5;

void setMotor(int MSpeed){
  cli();
  MSpeed = constrain(MSpeed, -50, 50);
  if(MSpeed >= 0){
    analogWrite(LMotor_PWM, 150 + MSpeed);
    digitalWrite(LMotor_DIG, LOW);
    analogWrite(RMotor_PWM, 150);
    digitalWrite(RMotor_DIG, LOW);
  }
  else{
    analogWrite(LMotor_PWM, 150);
    digitalWrite(LMotor_DIG, LOW);
    analogWrite(RMotor_PWM, 150 - MSpeed);
    digitalWrite(RMotor_DIG, LOW);
 }
 sei();
}

void setup() {
  Serial.begin(9600);
  pinMode(LMotor_PWM, OUTPUT);
  pinMode(LMotor_DIG, OUTPUT);
  pinMode(RMotor_PWM, OUTPUT);
  pinMode(RMotor_DIG, OUTPUT);
}

void loop() {
  if (Serial.available() > 0){
      String result = Serial.readStringUntil('\n');
//      int motorPower = result.toInt();
      Serial.println(result);
//      Serial.println(result);
//      setMotor(motorPower);
    }
}
