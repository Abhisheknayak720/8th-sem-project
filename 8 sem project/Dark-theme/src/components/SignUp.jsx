import React, {useState} from 'react';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider, responsiveFontSizes } from '@mui/material/styles';
import Link from '@mui/material/Link';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Copyright(props) {
  return (
    <Typography variant="body2" color="text.secondary" align="center" {...props}>
      {'Copyright © '}
      <Link color="inherit" href="https://mui.com/">
        Paintora
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>)
}

// Create a dark theme
let darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#f875aa', 
    },
    secondary: {
      main: '#f875aa', 
    },
  },
  typography: {
    fontSize: 20,
  },
});

// Enable responsive font sizes for the dark theme
darkTheme = responsiveFontSizes(darkTheme);

export default function SignUp() {

  

  const [fname, setFname] = useState("");
  const [lname, setLname] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPwd] = useState("");
  const [signupMessage, setSignupMessage] = useState("");

  const navigate = useNavigate();

  const setFnameHandler = (e) => {
      setFname(e.target.value);
  }

  const setLnameHandler = (e) => {
      setLname(e.target.value);
  }

  const setEmailHandler = (e) => {
      setEmail(e.target.value);
  }

  const setPwdHandler = (e) => {
      setPwd(e.target.value);
  }

  async function addSignUpData(e) {
    e.preventDefault();



    // Retrieve form data
    const formData = new FormData();
    formData.append("fname", fname);
    formData.append("lname", lname);
    formData.append("email", email); 
    formData.append("password", password);


    const config = {
        headers: {
            "Content-Type": "multipart/form-data"
        }
    };

    try {
        // Make POST request
        const res = await axios.post("/signup", formData, config);
        if(res.data.message === "Email already exists, redirecting to sign in..."){
          setTimeout(() => {
            navigate("/signin");
          }, 2000);
        }
        else if (res.status === 200) {
            console.log("Form submitted successfully!");
            // Redirect user if needed
            setSignupMessage(res.data.message);
            setTimeout(() => {
              navigate(`/interests?email=${encodeURIComponent(email)}`);
            }, 2000);
          }
        
       else if(res.data.message === "Error checking email existence. Try again after sometime"){
          setTimeout(()=>{
            navigate("/signin");
          },3000);
        }
       else {
            console.log("Error:", res.data);
        }
    } catch (error) {
        console.error("An error occurred:", error);
    }
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 15,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
            <LockOutlinedIcon />
          </Avatar>
          <Typography component="h1" variant="h5">
            Sign up
          </Typography>
          <Box component="form" noValidate onSubmit={addSignUpData} sx={{ mt: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  autoComplete="given-name"
                  name="firstName"
                  required
                  fullWidth
                  id="firstName"
                  label="First Name"
                  autoFocus
                  onChange={setFnameHandler}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  required
                  fullWidth
                  id="lastName"
                  label="Last Name"
                  name="lastName"
                  autoComplete="family-name"
                  onChange={setLnameHandler}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  autoComplete="email"
                  onChange={setEmailHandler}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  required
                  fullWidth
                  name="password"
                  label="Password"
                  type="password"
                  id="password"
                  onChange={setPwdHandler}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={<Checkbox value="allowExtraEmails" color="primary" />}
                  label="Get updates via mail"
                />
              </Grid>
            </Grid>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              {/* <Link href="/interests" sx={{color: 'white'}}> */}
                Sign Up
              {/* </Link> */}
            </Button>
            <div id='success'>{signupMessage}</div>
            <Grid container justifyContent="flex-end">
              <Grid item>
                <Link href="/signin">
                  Already have an account? Sign in
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
        <Copyright sx={{ mt: 5 }} />
      </Container>
    </ThemeProvider>
  );
}
