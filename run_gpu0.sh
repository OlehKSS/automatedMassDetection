#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

rm ./pom.xml
cp ./configurations/pom-gpu.xml pom.xml

# building and running
mvn compile
mvn exec:java -Dexec.mainClass=it.unicas.App
