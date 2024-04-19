const mysql = require("mysql");

const express = require("express");
const bodyParser = require("body-parser");
const cors =require('cors');

const app = express();
app.use(cors());



app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));
app.use(express.json());
app.use(bodyParser.json());

var db = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "drushti",
    database: "bruno",
    
});


var flag1 = 0;
  
db.connect(function(err) {
    if (err) {
        console.log(err);
    } else {
        console.log("Connected!");
        db.query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'bruno';",(err,tables)=>{
            for(var i=0;i<tables.length;i++){
                if(tables[i].TABLE_NAME == "users") flag1 = 1;
            }

            if(!flag1){
                var sql = "CREATE TABLE users (name varchar(30),email varchar(50),passwd text);"
                db.query(sql,function(err,result){
                    if(err) console.log(err);
                    else{
                        console.log("users created");
                    }
                });
            }         
        });   
    }
});

app.get("/xyz",(req,res)=>{
    res.json({message:"hello ak"});
});

app.post('/api/sendData', (req, res) => {
  const inputData = req.body.inputData;
  console.log('Data received from frontend:', inputData);
  res.json({ message: 'Data received successfully' });
});

app.listen(8000,function(){
    console.log('server started on port 8000');
});