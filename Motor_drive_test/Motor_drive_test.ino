 # define LMotor_LPWM 2
 # define LMotor_RPWM 3
 # define RMotor_RPWM 4
 # define RMotor_LPWM 5
 # define LMotor_LEN 22
 # define LMotor_REN 23
 # define RMotor_LEN 24
 # define RMotor_REN 25

 void setup(){
  pinMode(LMotor_LPWM, OUTPUT);
  pinMode(LMotor_RPWM, OUTPUT);
  pinMode(RMotor_RPWM, OUTPUT);
  pinMode(RMotor_LPWM, OUTPUT);
  pinMode(LMotor_LEN, OUTPUT);
  pinMode(LMotor_REN, OUTPUT);
  pinMode(RMotor_LEN, OUTPUT);
  pinMode(RMotor_REN, OUTPUT);

  // Pull the LEN to high volatage
  digitalWrite(LMotor_LEN, HIGH);
  digitalWrite(LMotor_REN, HIGH);
  digitalWrite(RMotor_REN, HIGH);
  digitalWrite(RMotor_LEN, HIGH);
 }

void loop(){
  // Moving Forward
  analogWrite(LMotor_LPWM, 100);
  analogWrite(LMotor_RPWM, 0);
  analogWrite(RMotor_RPWM, 100);
  analogWrite(RMotor_LPWM, 0);
}
