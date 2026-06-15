import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [symptoms, setSymptoms] = useState({ age: 45, sex: 1, bilirubin: 1.2, platelets: 250 });

  const uploadAndPredict = async () => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("symptoms_json", JSON.stringify(symptoms));

    const response = await axios.post("http://localhost:8000/predict", formData);
    setResult(response.data);
  };

  return (
    <div style={{ padding: '50px', fontFamily: 'Arial' }}>
      <h1>Liver Cirrhosis AI Assistant</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <h3>Upload Ultrasound/CT Scan</h3>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      </div>

      <button onClick={uploadAndPredict} style={{ padding: '10px 20px', background: 'blue', color: 'white' }}>
        Analyze Health
      </button>

      {result && (
        <div style={{ marginTop: '30px', border: '1px solid green', padding: '20px' }}>
          <h2>Diagnosis: {result.diagnosis}</h2>
          <p>Confidence: {result.confidence}</p>
          <p><b>Advice:</b> {result.recommendation}</p>
        </div>
      )}
    </div>
  );
}

export default App;