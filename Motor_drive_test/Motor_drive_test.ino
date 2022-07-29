 # define LLPWM 2
 # define LRPWM 3
 # define RRPWM 4
 # define RLPWM 5
 # define LLEN 22
 # define LREN 23
 # define RLEN 24
 # define RREN 25

 void setup(){
  pinMode(LLPWM, OUTPUT);
  pinMode(LRPWM, OUTPUT);
  pinMode(RRPWM, OUTPUT);
  pinMode(RLPWM, OUTPUT);
  pinMode(LLEN, OUTPUT);
  pinMode(LREN, OUTPUT);
  pinMode(RLEN, OUTPUT);
  pinMode(RREN, OUTPUT);
  
  digitalWrite(LLEN, HIGH);
  digitalWrite(LREN, HIGH);
  digitalWrite(RREN, HIGH);
  digitalWrite(RLEN, HIGH);
 }

void loop(){
  analogWrite(LLPWM, 200);
  analogWrite(LRPWM, 0);
  analogWrite(RRPWM, 0);
  analogWrite(RLPWM, 200);
}
