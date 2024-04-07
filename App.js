import React, { useState, useEffect } from "react";
import { Text, View, SafeAreaView, StyleSheet, Image } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { Asset } from "expo-asset";
// import * as tf from '@tensorflow/tfjs-core';
// import '@tensorflow/tfjs-backend-webgl';
import {
  bundleResourceIO,
  decodeJpeg,
} from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";
import Header from "./components/Header";
const MODEL_JSON = require("./assets/model.json");
const MODEL_WEIGHTS_1 = require("./assets/group1-shard1of3.bin");
const MODEL_WEIGHTS_2 = require("./assets/group1-shard2of3.bin");
const MODEL_WEIGHTS_3 = require("./assets/group1-shard3of3.bin");

const labels = {
  1: { name: "D00", id: 1 },
  2: { name: "D01", id: 2 },
  3: { name: "D10", id: 3 },
  4: { name: "D11", id: 4 },
  5: { name: "D20", id: 5 },
  6: { name: "D40", id: 6 },
  7: { name: "D43", id: 7 },
  8: { name: "D44", id: 8 },
  9: { name: "D50", id: 9 },
  10: { name: "person", id: 10 },
  11: { name: "bicycle", id: 11 },
  12: { name: "car", id: 12 },
  13: { name: "motorcycle", id: 13 },
  14: { name: "bus", id: 14 },
  15: { name: "truck", id: 15 },
  16: { name: "accident", id: 16 },
};

var imgUrl ="./assets/car180.jpg"

const App = () => {
  const [filteredPredictions, setFilteredPredictions] = useState(null);
  const [model, setModel] = useState(null);
  const [imageTensor, setImageTensor] = useState(null);


  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        
        const Model = await tf.loadGraphModel(
          bundleResourceIO(MODEL_JSON, [
            MODEL_WEIGHTS_1,
            MODEL_WEIGHTS_2,
            MODEL_WEIGHTS_3,
          ])
        );
        setModel(Model);
        console.log("Model loaded");
      } catch (error) {
        console.error("Error loading model: ", error);
      }
    };
    const loadImageTensor = async () => {
      try {
        const imageAsset = Asset.fromModule(
          require(imgUrl)
        );
        await imageAsset.downloadAsync(); // Ensure the asset is downloaded
        const imageUri = imageAsset.localUri;

        const imageDataArrayBuffer = await FileSystem.readAsStringAsync(
          imageUri,
          {
            encoding: FileSystem.EncodingType.Base64,
          }
        );
        const imageData = new Uint8Array(
          Buffer.from(imageDataArrayBuffer, "base64")
        );

        // Decode image data to a tensor
        const ImageTensor = decodeJpeg(imageData);
        setImageTensor(ImageTensor);
        console.log("Image tensor ready");
      } catch (error) {
        console.error("Error loading image tensor: ", error);
      }
    };
    loadModel();
    loadImageTensor();
  }, []);

  // Run prediction when filteredPredictions change
  useEffect(() => {
    const predict = async () => {
      if (!model || !imageTensor) return; // Ensure model and image tensor are loaded

      try {
        // Make prediction using executeAsync
        const predictions = await model.executeAsync(imageTensor.expandDims(0));
        const classes = await predictions[3].array();
        const scores = await predictions[2].array();
        const boxes = await predictions[1].array();

        const threshold = 0.4;
        const combinedArray = classes[0]
          .map((_, index) => ({
            box: boxes[0][index],
            score: scores[0][index],
            class: labels[classes[0][index]].name,
          }))
          .filter((item) => item.score > threshold);
        setFilteredPredictions(combinedArray);
        console.log(combinedArray);
      } catch (error) {
        console.error("Error during prediction: ", error);
      }
    };

    predict();
  }, [ model, imageTensor]);



  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.innerContainer}>
        <Header />
        <Image
          style={styles.img}
          source={require(imgUrl)} // Change the path to your logo image
          resizeMode="contain"
        />
        <Text>{filteredPredictions?JSON.stringify(filteredPredictions):'loading'}</Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1, // Ensure that SafeAreaView takes up the entire screen
  },
  innerContainer: {
    display:'flex', // Ensure that the inner container takes up the entire SafeAreaView
  },
  img: {
    padding: 0,
    // flex: 1,
    width: "100%",
    alignItems: "center",
    backgroundColor: "#0553",
    display: "flex",
    
  },
});

export default App;
