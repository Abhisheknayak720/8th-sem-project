// import React, { useState } from "react";
// import styled from "styled-components";
// import { Button } from "@mui/material";

// const Container = styled.div`
//   position: absolute;
//   top: 50%;
//   left: 50%;
//   transform: translate(-50%, -50%);
//   display: flex;
//   flex-direction: column;
//   align-items: center;
//   justify-content: center;
//   width: 68rem;
//   height: 39rem;
//   background-color: #f0f0f0; /* Background color */
//   border-radius: 20px; /* Rounded corners */
// `;


// const ButtonContainer = styled.div`
//   display: flex;
//   flex-direction: column;
//   align-items: center;
// `;

// const UploadButton = styled(Button)`
//   margin-bottom: 10px;
// `;

// const SubmitButton = styled(Button)`
//   margin-top: 10px;
// `;

// function ImageUpload() {
//   const [file, setFile] = useState(null);

//   return (
//     <Container>
//       <ButtonContainer>
//         <UploadButton variant="contained" component="label">
//           Upload File
//           <input type="file" name="pic" hidden />
//         </UploadButton>
//         <SubmitButton variant="contained" type="submit">
//           Submit
//         </SubmitButton>
//       </ButtonContainer>
//     </Container>
//   );
// }

// export default ImageUpload;


// import React, { useState } from 'react'
// import Button from 'react-bootstrap/Button';
// import Form from 'react-bootstrap/Form';
// import axios from "axios"
// import {useNavigate} from "react-router-dom"

// const ImageUpload = () => {

//     const [fname,setFName] = useState("");
//     const [file,setFile] = useState("");

//     const history = useNavigate();

//     const setdata = (e)=>{
//         setFName(e.target.value)
//     }

//     const setimgfile = (e)=>{
//         setFile(e.target.files[0])
//     }

//     const addUserData = async(e)=>{
//         e.preventDefault();

//         var formData = new FormData();
//         formData.append("photo",file)
//         formData.append("fname",fname);

//         const config = {
//             headers:{
//                 "Content-Type":"multipart/form-data"
//             }
//         }

//         const res = await axios.post("/register",formData,config);
       
//         if(res.data.status == 201){
//             history("/")
//         }else{
//             console.log("error")
//         }
//     }

//     return (
//         <>
//             <div className='container mt-3'>
//                 <h1>Upload Your Img Here</h1>

//                 <Form>
//                     <Form.Group className="mb-3" controlId="formBasicEmail">
//                         <Form.Label>UserName</Form.Label>
//                         <Form.Control type="text" name='fname' onChange={setdata} />
//                     </Form.Group>

//                     <Form.Group className="mb-3" controlId="formBasicPassword">
//                         <Form.Label>Select Your Image</Form.Label>
//                         <Form.Control type="file" name='photo' onChange={setimgfile} />
//                     </Form.Group>
//                     <Button variant="primary" type="submit" onClick={addUserData}>
//                         Submit
//                     </Button>
//                 </Form>
//             </div>
//         </>
//     )
// }

// export default ImageUpload 


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
              history("/");
          } else {
              console.log("Error:", res.data);
          }
      } catch (error) {
          console.error("Error:", error);
      }
  }

  return (
      <>
          <div className='container mt-3'>
              <h1>Upload Your Img Here</h1>

              <Form>
                  <Form.Group className="mb-3" controlId="formBasicEmail">
                      <Form.Label>UserName</Form.Label>
                      <Form.Control type="text" name='user_name' onChange={setUserNameHandler} />
                  </Form.Group>

                  <Form.Group className="mb-3" controlId="formBasicPassword">
                      <Form.Label>Select Your Image</Form.Label>
                      <Form.Control type="file" name='photo' onChange={setImgFileHandler} />
                  </Form.Group>
                  <Button variant="primary" type="submit" onClick={addUserData}>
                      Submit
                  </Button>
              </Form>
          </div>
      </>
  )
}

export default ImageUpload;
