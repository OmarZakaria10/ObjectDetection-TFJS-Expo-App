import React, { useState, useEffect, useRef } from "react";
import {
  Text,
  View,
  SafeAreaView,
  StyleSheet,
  Image,
  Dimensions,
  Platform,
  LogBox,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import { Asset } from "expo-asset";
import { Camera } from "expo-camera";
import Canvas from "react-native-canvas";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";
import Header from "./components/Header";
const MODEL_JSON = require("./assets/model.json");
const MODEL_WEIGHTS_1 = require("./assets/group1-shard1of3.bin");
const MODEL_WEIGHTS_2 = require("./assets/group1-shard2of3.bin");
const MODEL_WEIGHTS_3 = require("./assets/group1-shard3of3.bin");

const TensorCamera = cameraWithTensors(Camera);
const { width, height } = Dimensions.get("window");
LogBox.ignoreAllLogs(true);

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

var imgUrl = "./assets/car180.jpg";

const App = () => {
  const [filteredPredictions, setFilteredPredictions] = useState(null);
  const [model, setModel] = useState(null);
  const [imageTensor, setImageTensor] = useState(null);
  // const cameraRef = useRef(null);
  // const context = useRef(null);
  // const canvas = useRef(null);

  // const TensorCamera = cameraWithTensors(Camera);

  let textureDims;
  Platform.OS === "ios"
    ? (textureDims = { height: 1920, width: 1080 })
    : (textureDims = { height: 1200, width: 1600 });

  const predict = async (img) => {
    // if (!model || !imageTensor) return; // Ensure model and image tensor are loaded

    try {
      // Make prediction using executeAsync
      const predictions = await model.executeAsync(img.expandDims(0));
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

  const handleCameraStream =  (images) => {
    const loop = async () => {
      console.log(1)
        const nextImageTensor = images.next().value;
        console.log(2)
        if (nextImageTensor) {
          console.log(3)
          const objects = await predict(nextImageTensor);
          console.log(4)
          // console.log(objects.map((object) => object.className));
          console.log(objects);
          tf.dispose([nextImageTensor]);
        }
      
      requestAnimationFrame(loop);
    };
    loop();
  };

  useEffect(() => {
    const loadModel = async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        await tf.ready();
        tf.getBackend();
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
        const imageAsset = Asset.fromModule(require(imgUrl));
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
  }, []);

  return (
    <TensorCamera
      style={styles.camera}
      type={Camera.Constants.Type.back}
      onReady={handleCameraStream}
      resizeHeight={200}
      resizeWidth={152}
      resizeDepth={3}
      autorender={true}
      cameraTextureHeight={textureDims.height}
      cameraTextureWidth={textureDims.width}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  camera: {
    width: "100%",
    height: "100%",
  },
});
export default App;
