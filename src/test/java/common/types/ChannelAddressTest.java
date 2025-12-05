package common.types;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import common.exceptions.OpenemsError.OpenemsNamedException;

public class ChannelAddressTest {

    @Test
    void testValidChannelAddress() {
        String componentId = "meter0";
        String channelId = "ActivePower";
        ChannelAddress address = new ChannelAddress(componentId, channelId);
        assertEquals(componentId, address.getComponentId());
        assertEquals(channelId, address.getChannelId());
        assertEquals("meter0/ActivePower", address.toString());
    }

    @Test
    void testFromStringValid() throws OpenemsNamedException {
        String addressString = "meter0/ActivePower";
        ChannelAddress address = ChannelAddress.fromString(addressString);
        assertEquals("meter0", address.getComponentId());
        assertEquals("ActivePower", address.getChannelId());
        assertEquals(addressString, address.toString());
    }

    @Test
    void testFromStringInvalid() {
        assertThrows(OpenemsNamedException.class, () -> ChannelAddress.fromString("meter0"));
        assertThrows(OpenemsNamedException.class, () -> ChannelAddress.fromString("meter0/ActivePower/Extra"));
        assertThrows(OpenemsNamedException.class, () -> ChannelAddress.fromString(""));
    }

    @Test
    void testEqualsAndHashCode() {
        ChannelAddress address1 = new ChannelAddress("c1", "ch1");
        ChannelAddress address2 = new ChannelAddress("c1", "ch1");
        ChannelAddress address3 = new ChannelAddress("c2", "ch2");

        assertTrue(address1.equals(address2));
        assertTrue(address2.equals(address1));
        assertEquals(address1.hashCode(), address2.hashCode());

        assertFalse(address1.equals(address3));
        assertNotEquals(address1.hashCode(), address3.hashCode());
        
        assertFalse(address1.equals(null));
        assertFalse(address1.equals("c1/ch1"));
    }
    
    @Test
    void testCompareTo() {
        ChannelAddress address1 = new ChannelAddress("a", "a");
        ChannelAddress address2 = new ChannelAddress("a", "b");
        ChannelAddress address3 = new ChannelAddress("b", "a");

        assertTrue(address1.compareTo(address2) < 0);
        assertTrue(address2.compareTo(address1) > 0);
        assertTrue(address3.compareTo(address2) > 0);
        assertEquals(0, address1.compareTo(new ChannelAddress("a", "a")));
    }
}
