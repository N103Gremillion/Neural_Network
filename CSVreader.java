
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class CSVreader {

  String filePath;
  NeuralNetwork csvNetwork;
  
  public CSVreader (String filePath) {
    this.filePath = filePath;
  }

  public String readCSV() {
    StringBuilder fileContents = new StringBuilder();

    // try to read current filePath file
    try (BufferedReader reader = new BufferedReader(new FileReader(this.filePath))) {
      String line;

      // While there is a line to read from
      while ((line = reader.readLine()) != null) {
        fileContents.append(line);
        fileContents.append('\n');
      }
    }

    catch (IOException error) {
      error.printStackTrace();
    }

    return fileContents.toString();
    
  }

}