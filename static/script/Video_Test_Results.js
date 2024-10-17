fetch('video_results')
  .then(response => response.json())
  .then(data => {
    // Log the retrieved data
    console.log(data);

    // Variable to store the total similarity score
    let totalSimilarityScore = 0;

    // Create Doughnut Chart
    var ctx = document.getElementById("myChart").getContext("2d");
    var myChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'no_face'],
        datasets: [{
          data: data[1],
          backgroundColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)',
            'rgba(145, 195, 219, 1)',
            'rgba(180, 221, 220, 1)'
          ],
          hoverOffset: 4
        }]
      },
      options: {
        title: {
          display: true,
          text: 'Emotion Analysis',
          responsive: true,
          maintainAspectRatio: false,
        }
      }
    });

    // Create Questions and Answers Table
    var tableContainer = document.getElementById("myTable");
    var table = document.createElement("table");

    // Table Headings
    var headingsRow = document.createElement("tr");
    var headings = ['Question', 'Preferred Answer', 'Your Answer', 'Similarity'];
    headings.forEach(headingText => {
      var heading = document.createElement("th");
      heading.appendChild(document.createTextNode(headingText));
      headingsRow.appendChild(heading);
    });
    table.appendChild(headingsRow);

    // Table Rows
    data[0].forEach(rowData => {
      var row = document.createElement("tr");
      rowData.forEach((cellData, index) => {
        var cell = document.createElement("td");
        var cellText = '';
        if (index === 3) {
          var similarity = parseFloat(cellData);
          var similarityPercentage = (similarity * 100).toFixed(2) + '%';
          cellText = document.createTextNode(similarityPercentage);
          // Aggregate similarity scores
          totalSimilarityScore += similarity;
        } else {
          cellText = document.createTextNode(cellData);
        }
        cell.appendChild(cellText);
        row.appendChild(cell);
      });
      table.appendChild(row);
    });
    tableContainer.appendChild(table);

    // Calculate the average similarity score
    var averageSimilarityScore = totalSimilarityScore / data[0].length;
    console.log("avg score:", averageSimilarityScore);
    document.getElementById("averageScore").textContent = averageSimilarityScore.toFixed(2); // Displaying with 2 decimal places

    // Create Suggestions/Improvements Table
    var suggestionsContainer = document.getElementById("suggestionsTable");
    var suggestionsTable = document.createElement("table");

    // Suggestions Table Headings
    var suggestionsHeadingsRow = document.createElement("tr");
    var suggestionsHeadings = ['Question', 'Suggestions for Improvement'];
    suggestionsHeadings.forEach(headingText => {
      var heading = document.createElement("th");
      heading.appendChild(document.createTextNode(headingText));
      suggestionsHeadingsRow.appendChild(heading);
    });
    suggestionsTable.appendChild(suggestionsHeadingsRow);

    // Suggestions Table Rows
    data[0].forEach(rowData => {
      var row = document.createElement("tr");
      var questionCell = document.createElement("td");
      questionCell.appendChild(document.createTextNode(rowData[0]));
      row.appendChild(questionCell);

      var suggestionCell = document.createElement("td");
      var similarity = parseFloat(rowData[3]);
      var suggestionText = 'Great job!';

      if (similarity < 0.5) {
        var suggestions = [
          'Consider elaborating more on your answers.',
          'Try providing more details to enrich your responses.',
          'Your answers could benefit from more elaboration.',
          'Your responses could be enhanced with further elaboration.',
          'Adding more detail would enrich your explanations.',
          'Consider offering additional information to deepen your insights.',
          'Providing more context would strengthen your points.',
          'Your answers would benefit from a more thorough explanation.',
          'Expanding on your ideas would add depth to your responses.',
          'Enriching your explanations with more detail would improve clarity.'
        ];
        suggestionText = suggestions[Math.floor(Math.random() * suggestions.length)];
      } else if (similarity < 0.8) {
        var suggestions = [
          'Good, but there’s room for improvement in your responses.',
          'You’re on the right track, but try to expand on your answers a bit more.',
          'Your responses are decent, but they could use some more depth.',
          'Nice effort, but you could add more details to make it stronger.',
          'You are getting there, but try to elaborate a bit more.',
          'This is a good start, but you can dig deeper into the topic.',
          'Well done, but there’s potential to expand your explanations.',
          'You’re doing well, but adding more examples would enhance your answers.',
          'Solid response, but it would benefit from a bit more insight.',
          'You are on the right path, but try to provide more comprehensive information.'
        ];
        suggestionText = suggestions[Math.floor(Math.random() * suggestions.length)];
      }

      var emotionData = data[1];
      if (emotionData[0] > 10) { // Example threshold for 'angry'
        suggestionText += ' Try to stay calm and composed.';
      }
      if (emotionData[4] > 10) { // Example threshold for 'sad'
        suggestionText += ' Maintain a positive demeanor.';
      }

      suggestionCell.appendChild(document.createTextNode(suggestionText));
      row.appendChild(suggestionCell);
      suggestionsTable.appendChild(row);
    });
    suggestionsContainer.appendChild(suggestionsTable);

    // Function to generate and download the report as PDF
    function generateAndDownloadPDF() {
      const { jsPDF } = window.jspdf;
      const doc = new jsPDF();

      // Draw the chart on the canvas and convert it to an image
      myChart.update(); // Ensure chart is updated before exporting
      var chartImage = myChart.toBase64Image();

      // Add the chart image to the PDF
      doc.setFontSize(16);
      doc.text('Emotion Analysis Report', 10, 10);
      doc.addImage(chartImage, 'PNG', 10, 20, 180, 100); // Adjust the dimensions and position as needed

      doc.setFontSize(12);
      doc.text('Questions and Answers:', 10, 130);
      data[0].forEach((rowData, index) => {
        var yPosition = 140 + index * 20;
        doc.text(`Question: ${rowData[0]}`, 10, yPosition);
        doc.text(`Preferred Answer: ${rowData[1]}`, 10, yPosition + 10);
        doc.text(`Your Answer: ${rowData[2]}`, 10, yPosition + 20);
        doc.text(`Similarity: ${(parseFloat(rowData[3]) * 100).toFixed(2)}%`, 10, yPosition + 30);
      });

      doc.text(`Score: ${averageSimilarityScore.toFixed(2)}`, 10, 250);

      doc.text('Suggestions for Improvement:', 10, 260);
      data[0].forEach((rowData, index) => {
        var similarity = parseFloat(rowData[3]);
        var suggestionText = 'Great job!';

        if (similarity < 0.5) {
          var suggestions = [
            'Consider elaborating more on your answers.',
            'Try providing more details to enrich your responses.',
            'Your answers could benefit from more elaboration.',
            'Your responses could be enhanced with further elaboration.',
            'Adding more detail would enrich your explanations.',
            'Consider offering additional information to deepen your insights.',
            'Providing more context would strengthen your points.',
            'Your answers would benefit from a more thorough explanation.',
            'Expanding on your ideas would add depth to your responses.',
            'Enriching your explanations with more detail would improve clarity.'
          ];
          suggestionText = suggestions[Math.floor(Math.random() * suggestions.length)];
        } else if (similarity < 0.8) {
          var suggestions = [
            'Good, but there’s room for improvement in your responses.',
            'You’re on the right track, but try to expand on your answers a bit more.',
            'Your responses are decent, but they could use some more depth.',
            'Nice effort, but you could add more details to make it stronger.',
            'You are getting there, but try to elaborate a bit more.',
            'This is a good start, but you can dig deeper into the topic.',
            'Well done, but there’s potential to expand your explanations.',
            'You’re doing well, but adding more examples would enhance your answers.',
            'Solid response, but it would benefit from a bit more insight.',
            'You are on the right path, but try to provide more comprehensive information.'
          ];
          suggestionText = suggestions[Math.floor(Math.random() * suggestions.length)];
        }

        var emotionData = data[1];
        if (emotionData[0] > 10) { // Example threshold for 'angry'
          suggestionText += ' Try to stay calm and composed.';
        }
        if (emotionData[4] > 10) { // Example threshold for 'sad'
          suggestionText += ' Maintain a positive demeanor.';
        }

        var yPosition = 270 + index * 20;
        doc.text(`Question: ${rowData[0]}`, 10, yPosition);
        doc.text(`Suggestion: ${suggestionText}`, 10, yPosition + 10);
      });

      doc.save('Emotion_Analysis_Report.pdf');
    }

    document.getElementById("downloadButton").addEventListener("click", generateAndDownloadPDF);
  });

