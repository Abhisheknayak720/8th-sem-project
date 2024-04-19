
import React, { useState } from 'react';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const ImageUpload = () => {

  const [userName, setUserName] = useState("");
  const [file, setFile] = useState("");

  const history = useNavigate();

  const setUserNameHandler = (e) => {
      setUserName(e.target.value);
  }

  const setImgFileHandler = (e) => {
      setFile(e.target.files[0]);
  }

  const addUserData = async (e) => {
      e.preventDefault();

      var formData = new FormData();
      formData.append("photo", file);
      formData.append("user_name", userName); 

      const config = {
          headers: {
              "Content-Type": "multipart/form-data"
          }
      }

      try {
          const res = await axios.post("/imageUpload", formData, config);
          if (res.status === 200) {
              history("/gallery");
          } else {
              console.log("Error:", res.data);
          }
      } catch (error) {
          console.error("Error:", error);
      }
  }

  return (
      <>
          {/* <div style={{display: 'flex',flexDirection: 'column',alignItems: 'center', marginTop: "8rem", backgroundImage: "url('https://images.unsplash.com/photo-1582555172866-f73bb12a2ab3?q=80&w=1780&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')",backgroundSize: 'contain'}} className='container mt-3'> */}
              <h1>Upload Your Art Here</h1>

              <div style={{padding: '11rem', backgroundImage: 'blueviolet',backgroundColor: 'black', borderRadius: '6rem'}}>

              <Form>
                  <Form.Group className="mb-3" controlId="formBasicEmail" style={{ marginBottom: "2rem" }}>
                      <Form.Label>UserName</Form.Label>
                      <Form.Control type="text" name='user_name' onChange={setUserNameHandler} />
                  </Form.Group>

                  <Form.Group className="mb-3" controlId="formBasicPassword" style={{ marginBottom: "2rem" }}>
                      <Form.Label>Select Your Image</Form.Label>
                      <Form.Control type="file" name='photo' onChange={setImgFileHandler} />
                  </Form.Group>
                  <Button variant="primary" type="submit" onClick={addUserData}>
                      Submit
                  </Button>
              </Form>
              </div>
          {/* </div> */}
      </>
  )
}

export default ImageUpload;