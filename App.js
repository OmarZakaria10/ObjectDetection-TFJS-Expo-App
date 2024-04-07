import React, { useState, useEffect } from "react";
import { Text, View, SafeAreaView, StyleSheet,Image } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { Asset } from "expo-asset";
import {
  bundleResourceIO,
  fetch,
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

const App = () => {
  const [isTfReady, setTfReady] = useState(false);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [filteredPredictions, setFilteredPredictions] = useState(null);

  useEffect(() => {
    const loadModelAndPredict = async () => {
      try {
        await tf.ready();
        const model = await tf.loadGraphModel(
          bundleResourceIO(MODEL_JSON, [
            MODEL_WEIGHTS_1,
            MODEL_WEIGHTS_2,
            MODEL_WEIGHTS_3,
          ])
        );
        console.log("Model loaded");

        const imageAsset = Asset.fromModule(
          require("./assets/Japan_013115.jpg")
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
        const imageTensor = decodeJpeg(imageData);
        console.log("Image tensor ready");

        // Make prediction using executeAsync
        const predictions = await model.executeAsync(imageTensor.expandDims(0));
        // setPredictions(predictions);
        console.log("Prediction done");
        // Assuming predictions is an array of tensors with the following order:
        // [numDetections, boxes, classes, scores]
        // const [numDetections, boxes, classes, scores] = predictions;

        const classes = await predictions[3].array();
        const scores = await predictions[2].array();
        const boxes = await predictions[1].array();
        //5
        // console.log(boxes)
        threshold = 0.4;
        const combinedArray = classes[0]
          .map((_, index) => ({
            box: boxes[0][index],
            score: scores[0][index],
            class: labels[classes[0][index]].name,
          }))
          .filter((item) => item.score > threshold);
        setFilteredPredictions(combinedArray);
        console.log(typeof filteredPredictions);
        console.log(filteredPredictions);
      } catch (error) {
        console.error("Error during model prediction: ", error);
      }
    };
    loadModelAndPredict();
  }, [isTfReady, model]);

  return (
    <SafeAreaView style={styles.container}>
      <View >
        <Header />
        <View style={styles.imgView}>
        <Image
          style={styles.img}
          source={require('./assets/Japan_013115.jpg')} // Change the path to your logo image
          resizeMode="contain"
        />
        </View>
        {/* <View>
        <Text>{JSON.stringify(filteredPredictions)}</Text>
        </View> */}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1, // Ensure that SafeAreaView takes up the entire screen
  },
  innerContainer: {
    flex: 1,
    display:'flex',

     // Ensure that the inner container takes up the entire SafeAreaView
  },
  img:{
    // padding :10,
    flex: 1,
    width: '100%',
    // alignItems: 'center',

  },
  imgView:{
    // padding :10,
    flex: 1,
    width: '100%',
    height:'100%',
    // alignItems: 'center',

  },

});

export default App;
