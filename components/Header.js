// Header.js

import React from 'react';
import { View, Image, StyleSheet, StatusBar, Text } from 'react-native';

const Header = () => {
  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#ffffff" barStyle="dark-content" />
      <View style={styles.content}>
        <Image
          style={styles.logo}
          source={require('../assets/TF_icon.png')} // Change the path to your logo image
          resizeMode="contain"
        />
        <Text style={styles.title}>
          TensorFlow 
        </Text>
        <Text style={styles.title2}>
          Lite
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    //paddingTop: StatusBar.currentHeight, // Ensure that the header starts below the status bar
    height: 20 + StatusBar.currentHeight, // Adjust the height to include status bar height
    direction: 'ltr',
    
    alignItems: 'center',
    backgroundColor: '#ffffff', // Change the background color as needed
    borderBottomWidth: 1, // Add a border if desired
    borderBottomColor: '#cccccc', // Change the border color as needed
  },
  content: {
    flexDirection: 'row', // Arrange children horizontally
    alignItems: 'center', // Align children vertically
    paddingHorizontal: 10, // Add some padding between the logo and text
  },
  logo: {
    width: 40, // Adjust the width as needed
    height: 40, // Adjust the height as needed
    marginRight: 10, // Add some margin to separate the logo from the text
  },
  title: {
    fontSize: 22, // Adjust the font size as needed
    fontWeight: 'bold', // Optionally, make the text bold
  },
  title2: {
    fontSize: 22, // Adjust the font size as needed
    fontWeight: 'bold', // Optionally, make the text bold
    color: '#fa7223'
  },
});

export default Header;
