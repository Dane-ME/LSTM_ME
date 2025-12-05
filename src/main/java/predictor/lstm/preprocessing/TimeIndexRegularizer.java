package predictor.lstm.preprocessing;

import predictor.lstm.data.TimeSeriesData;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TimeIndexRegularizer {

    public static TimeSeriesData regularize(ArrayList<OffsetDateTime> dates, ArrayList<Double> values, int intervalInMinutes) {
        if (dates == null || dates.isEmpty()) {
            return new TimeSeriesData(new ArrayList<>(), new ArrayList<>());
        }

        Map<OffsetDateTime, Double> dataMap = new HashMap<>();
        for (int i = 0; i < dates.size(); i++) {
            dataMap.put(dates.get(i), values.get(i));
        }

        ArrayList<OffsetDateTime> regularizedDates = new ArrayList<>();
        ArrayList<Double> regularizedValues = new ArrayList<>();

        OffsetDateTime cursor = dates.get(0);
        OffsetDateTime endTime = dates.get(dates.size() - 1);

        while (!cursor.isAfter(endTime)) {
            regularizedDates.add(cursor);
            regularizedValues.add(dataMap.getOrDefault(cursor, Double.NaN));
            cursor = cursor.plusMinutes(intervalInMinutes);
        }

        return new TimeSeriesData(regularizedDates, regularizedValues);
    }
}
