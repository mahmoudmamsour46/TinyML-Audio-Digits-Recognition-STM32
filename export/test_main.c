#include <stdio.h>
#include "nn_inference.h"
#include "test_sample.h"

int main(void) {
    int32_t logits[12];
    
    // Execute inference on test sample
    int predicted_class = nn_predict(sample_spectrogram, logits);
    
    // Print predicted result
    // On STM32, output can be routed via UART or watched in Live Expressions
    printf("Expected Class : %d (%s)\n", EXPECTED_CLASS, EXPECTED_CLASS_NAME);
    printf("Predicted Class: %d (%s)\n", predicted_class, CLASS_NAMES[predicted_class]);
    
    if (predicted_class == EXPECTED_CLASS) {
        printf("Test PASSED successfully!\n");
    } else {
        printf("Test FAILED!\n");
    }
    
    while (1) {
        // Main embedded loop
    }
    
    return 0;
}
