
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class CSVreader {

  String filePath;

  public CSVreader (String filePath) {
    this.filePath = filePath;
  }

  public static String readCSV(String csvPath) throws IOException {

        // formating the input using a string builder
        StringBuilder fileContents = new StringBuilder();
        BufferedReader reader = null;

        try {
          reader = new BufferedReader(new FileReader(csvPath));
          String line;
          int curLIneIndex = 0;

          // while there is a line to read from
          while ((line = reader.readLine()) != null){
              fileContents.append(line);
              fileContents.append('\n');
              curLIneIndex++;
          }
        }
        catch (IOException error) {
            error.printStackTrace();
        }
        finally {
          reader.close();
        }
        return fileContents.toString();
    }
}