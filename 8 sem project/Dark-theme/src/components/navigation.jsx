import React, { useState } from "react";
import { Link } from 'react-router-dom';
import axios from "axios";

export const Navigation = (props) => {  

  return (
    <nav id="menu" className="navbar navbar-default navbar-fixed-top">
      <div className="container">
        <div className="navbar-header">
          <button
            type="button"
            className="navbar-toggle collapsed"
            data-toggle="collapse"
            data-target="#bs-example-navbar-collapse-1"
          >
            {" "}
            <span className="sr-only">Toggle navigation</span>{" "}
            <span className="icon-bar"></span>{" "}
            <span className="icon-bar"></span>{" "}
            <span className="icon-bar"></span>{" "}
          </button>
            <span>
              <Link to="/" className="navbar-brand page-scroll">
                PAINTORA
              </Link>
            </span>
        </div>

        <div
          className="collapse navbar-collapse"
          id="bs-example-navbar-collapse-1"
        >
          <ul className="nav navbar-nav navbar-right">
            <li>
              <Link to="/about" className="page-scroll">
                About
              </Link>
            </li>
            <li>
              <Link to="/features" className="page-scroll">
                Features
              </Link>
            </li>
            <li>
              <Link to="/gallery" className="page-scroll">
                Gallery
              </Link>
            </li>
            <li>
              <a href="#team" className="page-scroll">
                Team
              </a>
            </li>
            <li>
              <a href="#testimonials" className="page-scroll">
                Testimonials
              </a>
            </li>
            <li>
              <a href="#contact" className="page-scroll">
                Contact
              </a>
            </li>
            <li>
              <Link to="/signin" className="page-scroll">
                Sign in
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};
