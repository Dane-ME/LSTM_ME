package predictor.lstm.common;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.InflaterInputStream;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import predictor.lstm.validator.ValidationSeasonalityModel;
import predictor.lstm.validator.ValidationTrendModel;
public class ReadAndSaveModels {
    protected static final String MODEL_FOLDER = File.separator + "lstm" + File.separator;

    private static final String MODEL_DIRECTORY = Paths.get("ems.data.dir")//
            .toFile()//
            .getAbsolutePath();

    public static void save(HyperParameters hyperParameters) {
        var modelName = hyperParameters.getModelName();
        var path = Paths.get(MODEL_DIRECTORY, MODEL_FOLDER, modelName);
        var directory = path.getParent().toFile();
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                System.err.println("Failed to create directory: " + directory);
                return;
            }
        }

        Gson gson = new GsonBuilder()//
                .registerTypeAdapter(OffsetDateTime.class, new OffsetDateTimeAdapter())//
                .create();

        try {
            var compressedData = compress(hyperParameters);
            var compressedDataString = Base64.getEncoder().encodeToString(compressedData);
            var json = gson.toJson(compressedDataString);

            try (FileWriter writer = new FileWriter(path.toFile())) {
                writer.write(json);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static HyperParameters read(String fileName) {
        String filePath = Paths.get(MODEL_DIRECTORY, MODEL_FOLDER, fileName)//
                .toString();
        System.out.println(filePath);
        try (Reader reader = new FileReader(filePath)) {
            Gson gson = new GsonBuilder()//
                    .registerTypeAdapter(OffsetDateTime.class, new OffsetDateTimeAdapter())//
                    .create();
            var json = gson.fromJson(reader, String.class);
            var deserializedData = Base64.getDecoder().decode(json);
            return decompress(deserializedData);
        } catch (IOException e) {
            var hyperParameters = new HyperParameters();
            hyperParameters.setModelName(fileName);
            return hyperParameters;
        }
    }

    public static byte[] compress(HyperParameters hyp) {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             DeflaterOutputStream dos = new DeflaterOutputStream(baos);
             ObjectOutputStream oos = new ObjectOutputStream(dos)) {

            oos.writeObject(hyp);
            dos.finish();
            return baos.toByteArray();

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static HyperParameters decompress(byte[] value) {
        HyperParameters hyperParameters = null;
        try (ByteArrayInputStream bais = new ByteArrayInputStream(value);
             InflaterInputStream iis = new InflaterInputStream(bais);
             ObjectInputStream ois = new ObjectInputStream(iis)) {
            hyperParameters = (HyperParameters) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return hyperParameters;
    }

    public static void adapt(HyperParameters hyperParameters, ArrayList<Double> data, ArrayList<OffsetDateTime> dates) {
        if (hyperParameters.getCount() == 0) {
            return;
        }

        var valSeas = new ValidationSeasonalityModel();
        var valTrend = new ValidationTrendModel();

        hyperParameters.resetModelErrorValue();

        valSeas.validateSeasonality(data, dates, hyperParameters.getAllModelSeasonality(), hyperParameters);
        valTrend.validateTrend(data, dates, hyperParameters.getAllModelsTrend(), hyperParameters);
    }
}
