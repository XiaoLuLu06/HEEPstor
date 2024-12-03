#include <stdio.h>
#include "drivers/fpu.hpp"
#include "gen/model.hpp"

int main(int argc, char* argv[]) {
    FloatingPointUnit::enable();

    // TODO: Disable heepstor assert

    printf("Model test");
}