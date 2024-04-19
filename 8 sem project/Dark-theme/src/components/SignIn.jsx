import React, { useState } from 'react';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Typography from '@mui/material/Typography';
import { createTheme, ThemeProvider, responsiveFontSizes } from '@mui/material/styles';
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

// Enable responsive font sizes for the dark theme
darkTheme = responsiveFontSizes(darkTheme);

export default function SignInSide() {
  const navigate = useNavigate();
  
  const [email, setEmail] = useState("");
  const [password, setPwd] = useState("");
  const [signinMessage, setSigninMessage] = useState("");

  const history = useNavigate();

  const setEmailHandler = (e) => {
      setEmail(e.target.value);
  }

  const setPwdHandler = (e) => {
      setPwd(e.target.value);
  }

  async function addSignInData(e) {
    e.preventDefault();

    // Retrieve form data
    const formData = new FormData();
    formData.append("email", email); 
    formData.append("password", password);


    const config = {
        headers: {
            "Content-Type": "multipart/form-data"
        }
    };

    try {
      const res = await axios.post("/signin", formData, config);
      if (res.status === 200) {
        console.log("Form submitted successfully!");
  
        // Redirect to the appropriate page based on response
        if (res.data.message === "Signin successful") {
          // Redirect to the desired page after successful sign-in
          setSigninMessage("Sign In successful. Redirecting to the Gallery...");
          setTimeout(()=>{
            navigate("/gallery");
          },1000);
       
        }
        else if(res.data.message === "Email not found, redirecting to sign up..."){
          setSigninMessage("Email not found, redirecting to sign up...");
          setTimeout(() => {
            navigate("/signup");
          }, 2000);
          
        } else {
          // Display error or other messages
          setSigninMessage(res.data.message);
        }
      } else {
        console.log("Error:", res.data);
        setSigninMessage(res.data.message || "An error occurred");
      }
    } catch (error) {
        console.error("An error occurred:", error);
    }
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <Grid container component="main" sx={{ height: '100vh' }}>
        <CssBaseline />
        <Grid
          item
          xs={false}
          sm={4}
          md={7}
          sx={{
            backgroundImage: 'url(https://source.unsplash.com/random?paintings)',
            backgroundRepeat: 'no-repeat',
            backgroundColor: (t) =>
              t.palette.mode === 'light' ? t.palette.grey[50] : t.palette.grey[900],
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
        <Grid item xs={12} sm={8} md={5} component={Paper} elevation={6} square>
          <Box
            sx={{
              my: 20,
              mx: 4,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
              <LockOutlinedIcon />
            </Avatar>
            <Typography component="h1" variant="h5">
              Sign in
            </Typography>
            <Box component="form" noValidate onSubmit={addSignInData} sx={{ mt: 1 }}>
              <TextField
                margin="normal"
                required
                fullWidth
                id="email"
                label="Email Address"
                name="email"
                autoComplete="email"
                autoFocus
                onChange={setEmailHandler}
              />
              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="password"
                autoComplete="current-password"
                onChange={setPwdHandler}
              />
              <FormControlLabel
                control={<Checkbox value="remember" color="primary" />}
                label="Remember me"
              />
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
              >
                Sign In
              </Button>
              <div style={{ color: 'red' }}>{signinMessage}</div>
              <Grid container>
                <Grid item xs>
                  <Link href="#" variant="body2">
                    Forgot password?
                  </Link>
                </Grid>
                <Grid item>
                  <Link href="/signup">
                    {"Don't have an account? Sign Up"}
                  </Link>
                </Grid>
              </Grid>
              <Copyright sx={{ mt: 5 }} />
            </Box>
          </Box>
        </Grid>
      </Grid>
    </ThemeProvider>
  );
}
