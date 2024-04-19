// Importing required modules
const express = require("express");
const app = express();
const mysql = require("mysql");
const bodyParser = require("body-parser");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");

// Set up view engine
app.set("view engine", "ejs");

// Middleware for parsing request bodies
app.use(bodyParser.urlencoded({ extended: true }));

// Creating MySQL connection
const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "Abhishek@18",
  database: "paintora"
});

// Flags to check if tables exist
let flag5 = 0;
let flag6 = 0;
let flag7 = 0;

// Checking database connection and creating tables if not exist
db.connect(function(err) {
  if (err) {
    console.log(err);
  } else {
    console.log("Database connected!");

    // Creating 'paintings' table if not exists
    db.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'paintora';`, (err, tables) => {
      for (let i = 0; i < tables.length; i++) {
        if (tables[i].TABLE_NAME == "paintings") flag5 = 1;
      }
      if (!flag5) {
        const sql = "CREATE TABLE paintings (id INT AUTO_INCREMENT PRIMARY KEY, user_name VARCHAR(255) NOT NULL,image MEDIUMBLOB NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, THEME VARCHAR(100));";
        db.query(sql, (err, result) => {
          if (err) console.log(err);
          else {
            console.log("PAINTINGS created");
          }
        });
      }
    });

    // Creating 'contact' table if not exists
    db.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'paintora';`, (err, tables) => {
      for (let i = 0; i < tables.length; i++) {
        if (tables[i].TABLE_NAME == "contact") flag6 = 1;
      }
      if (!flag6) {
        const sql = "CREATE TABLE contact (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(50), email VARCHAR(50), message VARCHAR(50));";
        db.query(sql, (err, result) => {
          if (err) console.log(err);
          else {
            console.log("CONTACT created!");
          }
        });
      }
    });

    // Creating 'user_cred' table if not exists
    db.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'paintora';`, (err, tables) => {
      for (let i = 0; i < tables.length; i++) {
        if (tables[i].TABLE_NAME == "user_cred") flag7 = 1;
      }
      if (!flag7) {
        const sql = "CREATE TABLE user_cred (id INT AUTO_INCREMENT PRIMARY KEY, fname VARCHAR(50), lname VARCHAR(50), uemail VARCHAR(50), pwd VARCHAR(50), interests VARCHAR(100));";
        db.query(sql, (err, result) => {
          if (err) console.log(err);
          else {
            console.log("USER_CRED created!");
          }
        });
      }
    });
  }
});

// Set up multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = 'uploads'; // Specify the directory where you want to store the uploaded files
    fs.mkdirSync(uploadPath, { recursive: true }); // Create the directory if it doesn't exist
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname); // Use the original file name for storing
  },
});

const upload = multer({ storage: storage });
var variable = false;

// Serving static files
app.use(express.static("public"));
app.use(cors());

// Home route
app.get("/", (req, res) => {
  res.send("helloooooo"); 
});

// Handling image upload
// app.post("/imageUpload", upload.single("photo"), (req, res) => {
//   const { user_name } = req.body;
//   const imageFile = req.file;

//   if (!imageFile) {
//     return res.status(400).json({ message: "No image uploaded" });
//   }

//   const imageContent = fs.readFileSync(imageFile.path);

//   const query = "INSERT INTO paintings (user_name, image) VALUES (?, ?)";
//   db.query(query, [user_name, imageContent], (err, result) => {
//     if (err) {
//       console.error("Error inserting image into database:", err);
//       return res.status(500).json({ message: "Error uploading image" });
//     }
//     console.log("Successfully inserted image into database!");
//     res.status(200).json({ message: "Image uploaded successfully" });
//   });
// });

const { spawn } = require('child_process');
const path = require('path');

app.post("/imageUpload", upload.single("photo"), (req, res) => {
  const { user_name } = req.body;
  const imageFile = req.file;

  if (!imageFile) {
    return res.status(400).json({ message: "No image uploaded" });
  }

  const imageContent = fs.readFileSync(imageFile.path);

  const query = "INSERT INTO paintings (user_name, image) VALUES (?, ?)";
  db.query(query, [user_name, imageContent], (err, result) => {
    if (err) {
      console.error("Error inserting image into database:", err);
      return res.status(500).json({ message: "Error uploading image" });
    }
    console.log("Successfully inserted image into database!");

    // Specify the full file path of test3.py
    const pythonScriptPath = './multiclass_classifier/test3.py';
    
    // Execute the Python script
    const pythonProcess = spawn('python', [pythonScriptPath]);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python script stdout: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python script stderr: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);
    });

    res.status(200).json({ message: "Image uploaded successfully" });
  });
});



// Route to get images
app.get("/getimages", (req, res) => {
  console.log(xxx);
  const sql = `
    SELECT image, user_name 
    FROM paintings 
    WHERE theme IN (
      SELECT TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(interests, ',', numbers.n), ',', -1)) AS theme
      FROM
      (SELECT 1 n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) numbers
      INNER JOIN user_cred ON interests <> '' AND uemail = ?
      WHERE CHAR_LENGTH(interests) - CHAR_LENGTH(REPLACE(interests, ',', '')) >= numbers.n - 1
    )`;

  db.query(sql, [xxx], (err, results) => {
    if (err) {
      console.error("Error fetching images:", err);
      return res.status(500).json({ message: "Error fetching images" });
    }

    const images = results.map(result => ({
      imageData: result.image.toString('base64'),
      user_name: result.user_name
    }));
    res.json(images);
  });
});
var xxx = ""

// Handling contact form submission
app.post('/contact', upload.single("photo"), (req, res) => {
  const { name, email, message } = req.body;

  const query = "INSERT INTO contact (name, email, message) VALUES (?, ?, ?)";
  db.query(query, [name, email, message], (err, result) => {
    if (err) {
      console.error("Error inserting contact data into database:", err);
      return res.status(500).json({ message: "Error uploading contact form" });
    }
    console.log("Successfully inserted contact data into database!");
    res.status(200).json({ message: "Data uploaded successfully" });
  });
});

//saving user interests

app.post('/interests', upload.single("photo"), (req, res) => {
  const { intr, email } = req.body;
  // Here you can save the email and interests to your database or perform any other necessary actions
  console.log('Received interests:', intr, 'from', email);
  xxx = email
  const query = "UPDATE user_cred SET interests = ? WHERE uemail = ?";
  db.query(query, [intr, email], (err, result) => {
    if (err) {
      console.error("Error inserting data into database:", err);
      return res.status(500).json({ message: "Error uploading form" });
    }
    console.log("Successfully inserted data into database!");
    res.status(200).json({ message: 'Interests saved successfully' });
  });
});




// Handling user signup
app.post("/signup", upload.single("photo"), (req, res) => {
  const { fname, lname, email, password} = req.body;
  
  // Checking if email already exists
  const checkQuery = "SELECT COUNT(*) AS count FROM user_cred WHERE uemail = ?";
  db.query(checkQuery, [email], (checkErr, checkResult) => {
    if (checkErr) {
      console.error("Error checking email existence in database:", checkErr);
      return res.status(500).json({ message: "Error checking email existence" });
    }
    
    if (checkResult[0].count > 0) {
      return res.json({ message: "Email already exists, redirecting to sign in..." });
    }

    // Inserting user data into database
    const insertQuery = "INSERT INTO user_cred (fname, lname, uemail, pwd) VALUES (?, ?, ?, ?)";
    db.query(insertQuery, [fname, lname, email, password], (insertErr, result) => {
      if (insertErr) {
        console.error("Error inserting data into database:", insertErr);
        return res.status(500).json({ message: "Error uploading form" });
      }
      variable = true;
      console.log("Successfully inserted data into database!");
      res.status(200).json({ message: "Signed up successfully" });
    });
  });
});

// Handling user signin
app.post("/signin", upload.single("photo"), (req, res) => {
  const { email, password } = req.body;
  xxx = email;

  const checkCredentialsQuery = "SELECT * FROM user_cred WHERE uemail = ?";
  db.query(checkCredentialsQuery, [email], (checkCredErr, checkCredResult) => {
    if (checkCredErr) {
      console.error("Error checking credentials:", checkCredErr);
      return res.json({ message: "Error checking credentials" });
    }

    if (checkCredResult.length === 0) {
      // Email doesn't exist, send appropriate message
      return res.json({ message: "Email not found, redirecting to sign up..." });
    }

    const user = checkCredResult[0];
    if (user.pwd !== password) {
      // Password doesn't match, send appropriate message
      return res.json({ message: "Incorrect password" });
    }

    // Email and password match, send success message or perform further actions
    variable = true;
    res.json({ message: "Signin successful" });
  });
});

app.post("/nav", upload.single("photo"), (req, res) => {
  return res.json({ msg: "logged in" });
})

// Set up server port
const PORT = process.env.PORT || 8080;

// Start server
app.listen(PORT, console.log(`Server started on port ${PORT}`));
