#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

rm ./pom.xml
cp ./configurations/pom-gpu.xml pom.xml

# building and running on maven
mvn compile
# -X
mvn exec:java -e -Dexec.mainClass=it.unicas.App

# running on java
#mvn clean install
#mvn verify

#java -cp ./target/automated-mass-detection-1.0.jar it.unicas.App
