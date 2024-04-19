import { useState } from "react";
import React from "react";
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Form from 'react-bootstrap/Form';
import Button from "react-bootstrap/esm/Button";

export const Contact = (props) => {




  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [x, setX] = useState("");

  const history = useNavigate();

  const setNameHandler = (e) => {
      setName(e.target.value);
  }

  const setEmailHandler = (e) => {
      setEmail(e.target.value);
  }

  const setMessageHandler = (e) => {
      setMessage(e.target.value);
  }

  async function addContactData(e) {
    e.preventDefault();

    // Retrieve form data
    const formData = new FormData();
    formData.append("name", name); // Assuming 'name', 'email', 'message' are obtained from form inputs
    formData.append("email", email); 
    formData.append("message", message);

    // Assuming 'setX' is a state setter function to update some state
    setX("Sent successfully");

    const config = {
        headers: {
            "Content-Type": "multipart/form-data"
        }
    };

    try {
        // Make POST request
        const res = await axios.post("/contact", formData, config);
        if (res.status === 200) {
            console.log("Form submitted successfully!");
            // Redirect user if needed
            setName("")
            setEmail("")
            setMessage("")
            history("/");
        } else {
            console.log("Error:", res.data);
        }
    } catch (error) {
        console.error("An error occurred:", error);
    }
  }
  



  
  return (
    <div>
      <div id="contact">
        <div className="container">
          <div className="col-md-8">
            <div className="row">
              <div className="section-title">
                <h2>Get In Touch</h2>
                <p>
                  Please fill out the form below to send us an email and we will
                  get back to you as soon as possible.
                </p>
              </div>
              {/* <form name="sentMessage"> */}
              <Form>
                <div className="row">
                  <div className="col-md-6">
                    <Form.Group>
                      <Form.Control
                        style={{borderRadius: '0.5rem'}}
                        type="text"
                        id="name"
                        name="name"
                        value={name}
                        // className="form-control"
                        placeholder="Name"
                        required
                        onChange={setNameHandler}
                        autoComplete="off"
                      />
                      <p className="help-block text-danger"></p>
                    </Form.Group>
                  </div>
                  <div className="col-md-6">
                    <Form.Group>
                      <Form.Control
                        style={{borderRadius: '0.5rem'}}
                        type="email"
                        id="email"
                        name="email"
                        value={email}
                        // className="form-control"
                        placeholder="Email"
                        required
                        onChange={setEmailHandler}
                        autoComplete="off"
                      />
                      <p className="help-block text-danger"></p>
                    </Form.Group>
                  </div>
                </div>
                <Form.Group>
                  <textarea
                    style={{borderRadius: '0.5rem', resize: 'none'}}
                    name="message"
                    id="message"
                    className="form-control"
                    rows="4"
                    placeholder="Message"
                    value={message}
                    required
                    onChange={setMessageHandler}
                  ></textarea>
                  <p className="help-block text-danger"></p>
                </Form.Group>
                <div id="success"><h1>{x}</h1></div>
                <Button type="submit" className="btn btn-custom btn-lg" onClick={addContactData}>
                  Send Message
                </Button>
                </Form>
            </div>
          </div> 
          <div className="col-md-3 col-md-offset-1 contact-info">
            <div className="contact-item">
              <h3>Contact Info</h3>
              <p>
                <span>
                  <i className="fa fa-map-marker"></i> Address
                </span>
                <a target="_blank" className="greyhover" href="https://maps.app.goo.gl/NVZWayQ7sHtuK71B6">{props.data ? props.data.address : "loading"}</a>
              </p>
            </div>
            <div className="contact-item">
              <p>
                <span>
                  <i className="fa fa-phone"></i> Phone
                </span>{" "}
                <a className="greyhover" href={props.data ? `tel:${props.data.ph1}` : "#"}>{props.data ? props.data.phone1 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `tel:${props.data.ph2}` : "#"}>{props.data ? props.data.phone2 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `tel:${props.data.ph3}` : "#"}>{props.data ? props.data.phone3 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `tel:${props.data.ph4}` : "#"}>{props.data ? props.data.phone4 : "loading"}</a>
              </p>
            </div>
            <div className="contact-item">
              <p>
                <span>
                  <i className="fa fa-envelope-o"></i> Email
                </span>{" "}
                <a className="greyhover" href={props.data ? `mailto:${props.data.email1}` : "#"}>{props.data ? props.data.email1 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `mailto:${props.data.email2}` : "#"}>{props.data ? props.data.email2 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `mailto:${props.data.email3}` : "#"}>{props.data ? props.data.email3 : "loading"}</a><br />
                <a className="greyhover" href={props.data ? `mailto:${props.data.email4}` : "#"}>{props.data ? props.data.email4 : "loading"}</a>
              </p>
            </div>
          </div>
          <div className="col-md-12">
            <div className="row">
              <div className="social">
                <ul>
                  <li>
                    <a href={props.data ? props.data.facebook : "/"}>
                      <i className="fa fa-facebook"></i>
                    </a>
                  </li>
                  <li>
                    <a href={props.data ? props.data.instagram : "/"}>
                      <i className="fa fa-instagram"></i>
                    </a>
                  </li>
                  <li>
                    <a href={props.data ? props.data.pinterest : "/"}>
                      <i className="fa fa-pinterest"></i>
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div id="footer" style={{backgroundColor: 'black', color: 'white'}}>
        <div className="container text-center">
          <p>
            &copy; {new Date().getFullYear()} Paintora, Inc. Made with 💜 by{" "}
            <a href="https://github.com/Abhisheknayak720/8th-sem-project" rel="nofollow">
              Team 7
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};
