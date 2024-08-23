//
// Created by polya on 8/23/24.
//

#ifndef POTENTIALFUNCTIONPROTOTYPE_HPP
#define POTENTIALFUNCTIONPROTOTYPE_HPP
#include <cstring> // For memcpy
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>

class potentialFunction
{
    //base class for potential function
public:
    virtual double operator()(const double * xVec, const double * yVec, const int& j) = 0;
    virtual void json2Coefs(const std::string &coefsStr)=0;
    virtual  void init()=0;
    virtual double potentialFull(const double * xVec, const double * yVec)=0;
    virtual double getLm() const = 0; //  method to get lm
    // virtual double get_eps() const = 0; //  method to get eps
    virtual ~ potentialFunction() {};

};

std::shared_ptr<potentialFunction>  createPotentialFunction(const std::string& funcName, const std::string &row) ;

#endif //POTENTIALFUNCTIONPROTOTYPE_HPP
