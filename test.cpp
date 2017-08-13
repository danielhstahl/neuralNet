#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <iostream>
#include "NeuralNetGraph.h"


TEST_CASE("Test Forward", "[NNGraph]"){
    std::vector<int> numNodesInLayer={2, 1};
    std::vector<double> x={.3, .5};
    auto nodes=nngraph::forward_prop_init(numNodesInLayer, x, [&](const auto input){
        return 1.0/(1.0+exp(-input));
    });
    auto printLayer=[&](int layer, bool showW){
        for(int i=0; i<nodes[layer].size();++i){
            std::cout<<"oij: "<<nodes[layer][i]->oij<<std::endl;
            if(showW){
                for(int j=0;j<nodes[layer][i]->forwardW.size();++j){
                    std::cout<<"w: "<<nodes[layer][i]->forwardW[j].w<<std::endl;
                }
            }
            
        }
    };
    printLayer(0, true);
    printLayer(1, true);
    printLayer(2, false);

    //REQUIRE(newton::zeros(squareTestV, deriv, guess, .00001, 20)==Approx(sqrt(2.0)));
}  
  