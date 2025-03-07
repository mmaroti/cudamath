#include <iostream>
#include "linapprox.hpp"

int main(int argc, char **argv)
{
    LinApprox16 tester;

    std::vector<uint16_t> graph = {0, 257, 470, 471, 1889, 1962, 1995, 2050, 2225, 2227, 2628, 2712, 2780, 3130, 3147, 3185, 3843, 3940, 3943, 5411, 5531, 5560,
                                   6007, 6021, 6130, 6715, 6812, 6823, 8540, 8607, 8643, 9231, 9446, 9449, 9570, 9647, 9677, 9760, 9951, 9983, 10033, 10201,
                                   10216, 10582, 10636, 10714, 11372, 11409, 11517, 11584, 11683, 11747, 12057, 12233, 12240, 13635, 13703, 13764, 13895,
                                   14011, 14076, 14409, 14475, 14530, 14856, 15094, 15102, 15155, 15193, 15210, 15646, 15823, 15825, 15953, 16020, 16069,
                                   16388, 16507, 16511, 17486, 17555, 17629, 17954, 17996, 18030, 20525, 20616, 20645, 21765, 21929, 21932, 22141, 22199,
                                   22218, 22354, 22433, 22515, 22828, 22878, 22898, 24692, 24726, 24802, 24936, 24962, 25066, 25685, 25790, 25835, 25906,
                                   25999, 26045, 26233, 26293, 26316, 27399, 27416, 27423, 28201, 28382, 28407, 29449, 29473, 29480, 30043, 30080, 30171,
                                   30726, 30920, 30926, 32063, 32216, 32231, 32526, 32560, 32574, 33375, 33444, 33531, 34343, 34503, 34528, 35443, 35465,
                                   35578, 35638, 35782, 35824, 36637, 36779, 36790, 37156, 37252, 37280, 37386, 37445, 37455, 38477, 38585, 38644, 41268,
                                   41281, 41333, 42518, 42543, 42553, 43051, 43078, 43117, 43310, 43457, 43503, 44395, 44430, 44517, 45866, 45968, 46010,
                                   46355, 46470, 46485, 46648, 46784, 46840, 47127, 47342, 47353, 47386, 47534, 47540, 47629, 47703, 47706, 48914, 48962,
                                   48976, 49425, 49446, 49463, 50021, 50096, 50133, 50287, 50330, 50421, 50534, 50610, 50644, 52496, 52705, 52721, 52810,
                                   52902, 52972, 53118, 53165, 53203, 53285, 53331, 53366, 55664, 55709, 55789, 56186, 56232, 56274, 56587, 56722, 56729,
                                   57116, 57184, 57212, 58388, 58506, 58526, 59148, 59265, 59277, 60725, 60823, 60834, 61757, 61780, 61801, 62040, 62140,
                                   62180, 62780, 62851, 62911, 64283, 64355, 64376, 64533, 64584, 64605};

    std::tuple<int, uint16_t, uint16_t> result = tester.evaluate(graph);
    std::cout << std::get<0>(result) << " " << std::get<1>(result) << " " << std::get<2>(result) << std::endl;
}
