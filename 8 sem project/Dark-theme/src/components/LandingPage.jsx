// LandingPage.jsx
import React, { useState, useEffect } from "react";
import { Navigation } from "./navigation";
import { Header } from "./header";

import { Features } from "./features";
import About from "./About"
// import { Services } from "./services";
// import Gallery1 from "./Gallery1";
import { Testimonials } from "./testimonials";
import { Team } from "./Team";
import { Contact } from "./contact";
import JsonData from "../../src/data/data.json";
import SmoothScroll from "smooth-scroll";
import "../../src/App.css"
// import SignUp from "./SignUp";
// import SignInSide from "./SignIn";
// import SubmitReview from "./SubmitReview";


export const scroll = new SmoothScroll('a[href*="#"]', {
  speed: 1000,
  speedAsDuration: true,
});


const LandingPage = () => {
  const [landingPageData, setLandingPageData] = useState({});
  
  
  useEffect(() => {
    setLandingPageData(JsonData);
  }, []);
  return (
    <div>
       <Navigation />
        <Header data={landingPageData.Header} />
        <About />
        <Features />
        {/* <Gallery data={landingPageData.Gallery} /> */}
        <Team data={landingPageData.Team} />
        <Testimonials data={landingPageData.Testimonials} />
        <Contact data={landingPageData.Contact} />
    </div>
  );
};

export default LandingPage;
