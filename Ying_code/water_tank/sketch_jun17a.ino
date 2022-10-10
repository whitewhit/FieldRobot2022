int Num=1; //可更改成影像辨識Flag
void setup() {
 pinMode(8,OUTPUT);
 Serial.begin(9600);
}

void loop() {
  //讀取序列埠傳入的字元
  Serial.println(Num);
  if(Serial.available()){
    Num=Serial.read();
    
  } 
  
  delay(10);
  
  if(Num==49){ //ASCII 1=49
    digitalWrite(8,LOW); //低電平觸發，LOW時繼電器觸發
  }
  else{
    digitalWrite(8,HIGH);
  }
}
