import React, { useState } from 'react';
import { Button, TextField, Typography, Container, Grid, CssBaseline, createTheme, ThemeProvider } from '@mui/material';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';

let darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#f875aa', // Violet color
    },
    secondary: {
      main: '#f875aa', // Violet color
    },
  },
  typography: {
    fontSize: 20,
  },
});

function SubmitReview() {
  const [name, setName] = useState('');
  const [review, setReview] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  const handleNameChange = (e) => {
    setName(e.target.value);
  };

  const handleReviewChange = (e) => {
    setReview(e.target.value);
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // You can handle form submission here
    // Access name, review, and selectedFile for further processing
    console.log('Name:', name);
    console.log('Review:', review);
    console.log('Selected File:', selectedFile);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="sm">
        <Typography variant="h4" gutterBottom sx={{textAlign: 'center', padding: '12rem 0 3.5rem', color: 'white'}}>
          Drop us a line
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                label="Name"
                variant="outlined"
                fullWidth
                value={name}
                onChange={handleNameChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Review"
                variant="outlined"
                fullWidth
                multiline
                rows={4}
                value={review}
                onChange={handleReviewChange}
              />
            </Grid>
            <Grid item xs={12}>
              <input
                accept="image/*"
                id="icon-button-file"
                type="file"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <label htmlFor="icon-button-file">
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<PhotoCameraIcon />}
                  component="span"
                >
                  Upload Photo
                </Button>
              </label>
              {selectedFile && <span>{selectedFile.name}</span>}
            </Grid>
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                fullWidth
              >
                Submit
              </Button>
            </Grid>
          </Grid>
        </form>
      </Container>
    </ThemeProvider>
  );
}

export default SubmitReview;
