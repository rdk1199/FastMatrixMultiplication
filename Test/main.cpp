
#include <iostream>

#include <matrix.h>

#include "benchmarks.h"

int main()
{
    std::vector<double> entries(
        {
            1.0, -1.0, 2.0,
            -2.0, 3.0, -3.0,
            4.0, -4.0, 5.0
        }
    );
    
    Matrix test_matrix(3, 3, entries);
    

    //test_add();
    test_multiply();

    return 0;
}
