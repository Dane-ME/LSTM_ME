package predictor.lstm.utilities;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class UtilityConversionTest {

    @Test
    void to2DArrayFromArrayListTest() {
        ArrayList<ArrayList<Double>> source = new ArrayList<>();
        source.add(new ArrayList<>(Arrays.asList(1.0, 2.0)));
        source.add(new ArrayList<>(Arrays.asList(3.0, 4.0)));

        double[][] expected = {{1.0, 2.0}, {3.0, 4.0}};
        assertArrayEquals(expected, UtilityConversion.to2DArray(source));
    }
    
    @Test
    void to1DArrayFromListTest() {
        List<Double> source = Arrays.asList(1.1, 2.2, 3.3);
        double[] expected = {1.1, 2.2, 3.3};
        assertArrayEquals(expected, UtilityConversion.to1DArray(source), 1e-9);
    }
    
    @Test
    void to1DArrayListFrom1DArrayTest() {
        double[] source = {1.1, 2.2, 3.3};
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(1.1, 2.2, 3.3));
        assertEquals(expected, UtilityConversion.to1DArrayList(source));
    }
    
    @Test
    void to1DArrayListFrom2DArrayListTest() {
        ArrayList<ArrayList<Double>> source = new ArrayList<>();
        source.add(new ArrayList<>(Arrays.asList(1.0, 2.0)));
        source.add(new ArrayList<>(Arrays.asList(3.0, 4.0)));
        
        ArrayList<Double> expected = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0));
        assertEquals(expected, UtilityConversion.to1DArrayList(source));
    }

    @Test
    void getMinIndexTest() {
        double[] arr = {3.5, 2.0, 5.1, 1.2, 4.8};
        assertEquals(3, UtilityConversion.getMinIndex(arr));
    }

    @Test
    void getMinIndexWithEmptyArrayTest() {
        double[] arr = {};
        assertThrows(IllegalArgumentException.class, () -> UtilityConversion.getMinIndex(arr));
    }

    @Test
    void getMinIndexWithNullArrayTest() {
        double[] arr = null;
        assertThrows(IllegalArgumentException.class, () -> UtilityConversion.getMinIndex(arr));
    }
}
