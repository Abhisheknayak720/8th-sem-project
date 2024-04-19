import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Box from '@mui/material/Box';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import ImageListItemBar from '@mui/material/ImageListItemBar';

const ImgCmp = () => {
  const [images, setImages] = useState([]);

  useEffect(() => {
    const fetchImages = async () => { 
      try {
        const response = await axios.get('/getimages');
        setImages(response.data);
      } catch (error) {
        console.error('Error fetching images:', error);
      }
    };

    fetchImages();
  }, []);

  const getRandomDimension = () => {
    const options = [
      { width: 315, height: 211 },
      { width: 315, height: 421 },
      { width: 315, height: 209 },
      { width: 315, height: 473 },
      { width: 315, height: 227 },
      { width: 315, height: 314 },
    ];
    const randomIndex = Math.floor(Math.random() * options.length);
    return options[randomIndex];
  };

  return (
    <Box sx={{ width: "103rem", height: "67rem", overflowY: 'scroll' }}>
      <ImageList variant="masonry" cols={3} gap={8}>
        {images.map((image, index) => {
          const { width, height } = getRandomDimension();
          console.log(image.user_name);
          return (
            <ImageListItem key={index}>
              <img
                src={`data:image/jpeg;base64,${image.imageData}`}
                alt={`Image ${index}`}
                style={{ width: `${width}px`, height: `${height}px`, objectFit: 'cover' }} // Remove margin and padding
              />
              <ImageListItemBar position="below" title={image.user_name} />
            </ImageListItem>
          );
        })}
      </ImageList>
    </Box>
  );
};

export default ImgCmp;
