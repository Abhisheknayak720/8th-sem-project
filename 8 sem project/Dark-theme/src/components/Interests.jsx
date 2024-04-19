import React, { useState } from 'react';
import axios from 'axios';
import { useLocation, useNavigate } from 'react-router-dom';

const Interests = () => {
  const location = useLocation();
  const history = useNavigate();
  const searchParams = new URLSearchParams(location.search);
  const email = searchParams.get('email');
  const [items, setItems] = useState([
    'Mountain',
    'Desert',
    'Nature',
    'Greenery',
    'Architecture',
    'Seascapes',
    'Forest',
  ]);

  const [selectedItems, setSelectedItems] = useState([]);

  const toggleItem = (item) => {
    if (selectedItems.includes(item)) {
      setSelectedItems(selectedItems.filter((selectedItem) => selectedItem !== item));
      console.log(`Deselected: ${item}`);
    } else {
      setSelectedItems([...selectedItems, item]);
      console.log(`Selected: ${item}`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("email", email); // Append email to form data
    formData.append("intr", selectedItems);
    const config = {
      headers: {
        "Content-Type": "multipart/form-data"
      }
    };
    try {
      const response = await axios.post('/interests', formData, config);
      console.log('Selected items posted successfully:', response.data);
      setTimeout(() => {
        history("/gallery");
      }, 2000);
    } catch (error) {
      console.error('Error posting selected items:', error);
    }
  };
  

  return (
    <div className="intr-container" style={{'padding': '100px'}}>
      <h2 className="intr-title">Select your interests:</h2>
      <div className="intr-items">
        {items.map((item) => (
          <label key={item} className="intr-checkbox">
            <input
              type="checkbox"
              checked={selectedItems.includes(item)}
              onChange={() => toggleItem(item)}
            />
            <span className="custom-checkbox"></span> {/* New span for custom checkbox */}
            {item}
          </label>
        ))}
      </div>
      <button className="intr-submit" onClick={handleSubmit}>Submit</button>
    </div>
  );
};

export default Interests;
