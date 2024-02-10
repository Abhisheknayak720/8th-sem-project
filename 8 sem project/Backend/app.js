// const express = require("express");
// const app = express();
// const mysql = require("mysql");
// const bodyParser = require("body-parser");
// const multer = require("multer");
// const cors = require("cors");
// const fs = require("fs");

// app.set("view engine", "ejs");

// app.use(bodyParser.urlencoded({ extended: true }));



// var db = mysql.createConnection({
//     host: "localhost",
//     user: "root",
//     password: "Abhishek@18",
//     database: "paintora"
// });

// var flag5 = 0

// db.connect(function(err) {
//   if (err) {
//       console.log(err);
//   } else {
//       console.log("Connected!");
//       db.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'paintora';`,(err,tables)=>{

//         for(var i=0;i<tables.length;i++){
//           if(tables[i].TABLE_NAME == "paintings") flag5 = 1;
//         }

//         if(!flag5){
//             var sql = "CREATE TABLE paintings (id INT AUTO_INCREMENT PRIMARY KEY, user_name VARCHAR(255) NOT NULL,image MEDIUMBLOB NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);";
//             db.query(sql,function(err,result){
//                 if(err) console.log(err);
//                 else{
//                     console.log("PAINTINGS created");
//                 }
//             });
//         }
//     });
//   }
// });

// // Set up multer for file uploads
// const storage = multer.diskStorage({
//   destination: function (req, file, cb) {
//     cb(null, "uploads"); // Specify the directory where you want to store the uploaded files
//   },
//   filename: function (req, file, cb) {
//     cb(null, file.originalname); // Use the original file name for storing
//   },
// });



// const upload = multer({ storage: storage });

// app.use(express.static("public"));
// app.use(cors());

// app.get("/", (req, res) => {
//   res.send("helloooooo"); 
// });

// app.post("/post", upload.single("file"), (req, res) => {
//   console.log("Connected to React");
//   res.redirect("/");
// });

// const PORT = process.env.PORT || 8080;

// app.listen(PORT, console.log(`Server started on port ${PORT}`));



const express = require("express");
const app = express();
const mysql = require("mysql");
const bodyParser = require("body-parser");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");

app.set("view engine", "ejs");

app.use(bodyParser.urlencoded({ extended: true }));

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "Abhishek@18",
  database: "paintora"
});

let flag5 = 0;

db.connect(function(err) {
  if (err) {
    console.log(err);
  } else {
    console.log("Connected!");
    db.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'paintora';`, (err, tables) => {
      for (let i = 0; i < tables.length; i++) {
        if (tables[i].TABLE_NAME == "paintings") flag5 = 1;
      }
      if (!flag5) {
        const sql = "CREATE TABLE paintings (id INT AUTO_INCREMENT PRIMARY KEY, user_name VARCHAR(255) NOT NULL,image MEDIUMBLOB NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);";
        db.query(sql, (err, result) => {
          if (err) console.log(err);
          else {
            console.log("PAINTINGS created");
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

app.use(express.static("public"));
app.use(cors());

app.get("/", (req, res) => {
  res.send("helloooooo"); 
});

// Handling image upload
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
    res.status(200).json({ message: "Image uploaded successfully" });
  });
});


// Add a route to fetch images from the database
// app.get("/images", (req, res) => {
//   const query = "SELECT * FROM paintings";
//   db.query(query, (err, results) => {
//     if (err) {
//       console.error("Error fetching images:", err);
//       return res.status(500).json({ message: "Error fetching images" });
//     }
//     res.status(200).json(results);
//   });
// });


const PORT = process.env.PORT || 8080;

app.listen(PORT, console.log(`Server started on port ${PORT}`));
