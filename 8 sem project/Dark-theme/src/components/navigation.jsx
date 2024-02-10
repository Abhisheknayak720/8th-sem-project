import React from "react";
import { Link } from 'react-router-dom';


export const Navigation = (props) => {
  // const [anchorElUser, setAnchorElUser] = React.useState(null);
  // const handleOpenUserMenu = (event) => {
  //   setAnchorElUser(event.currentTarget);
  // };
  // const handleCloseUserMenu = () => {
  //   setAnchorElUser(null);
  // };
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
          <a className="navbar-brand page-scroll" href="#page-top">
            <span>
              {/* <img src="img/paintora-logos.png" width={50} />&nbsp; */}
              PAINTORA
            </span>
          </a>{" "}
          {/* <img src="img/paintora-logos.png" width={50} /> &nbsp; */}
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
              <a href="#testimonials" className="page-scroll">
                Testimonials
              </a>
            </li>
            <li>
              <a href="#team" className="page-scroll">
                Team
              </a>
            </li>
            <li>
              <a href="#contact" className="page-scroll">
                Contact
              </a>
            </li>
            {/* <li>
              <AccountBoxIcon fontSize="large" />
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleClose}
              >
                <MenuItem onClick={handleClose}>Profile</MenuItem>
                <MenuItem onClick={handleClose}>My account</MenuItem>
              </Menu>
            </li> */}
          </ul>
        </div>
      </div>
    </nav>
  );
};
