import React from "react";
import { Link } from "react-router-dom";
import styled from "styled-components";
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import ImgCmp from "./ImgComp";

const StyledCardContainer = styled.div`
  margin-top: 20px; /* Adjust margin as needed */
`;

const UploadButtonContainer = styled.div`
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
`;

const Gallery = (props) => {
  return (
    <div id="portfolio" className="text-center">
      <div className="container">
        <div className="section-title">
          <h2 style={{ color: "white" }}>Gallery</h2>
          <p style={{ color: "white" }}>
            Discover premier collections by renowned profiles
          </p>
        </div>
        <StyledCardContainer>
          <GridExample />
          <ImgCmp />
        </StyledCardContainer>
        <UploadButtonContainer>
          <Link
            to="/imageUpload"
            style={{ marginTop: "3rem", fontSize: "1.5rem", backgroundColor: "blue", borderRadius: "1rem", color: "white", padding: "1rem" }}
          >
            Upload Image
          </Link>
        </UploadButtonContainer>
      </div>
    </div>
  );
};

function GridExample() {
  // Define an array of image URLs
  const images = [
    "https://picsum.photos/200",
    "https://picsum.photos/200",
    "https://picsum.photos/200",
    "https://picsum.photos/200"
    // Add more image URLs as needed
  ];

  return (
    <Row xs={1} md={2} lg={4} className="g-4">
      {images.map((imageUrl, idx) => (
        <Col key={idx}>
        <div className='d-flex justify-content-between align-items-center mt-5'>
          <Card>
            <Card.Img variant="top" src={imageUrl} />
            <Card.Body>
              <Card.Title>Card {idx + 1}</Card.Title>
              <Card.Text>
                Some quick example text to build on the card title and make up the bulk of the card's content.
              </Card.Text>
            </Card.Body>
          </Card>
          </div>
        </Col>
      ))}
    </Row>
  );
}

export default Gallery;
