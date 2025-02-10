#include <cstdint>
#include <vector>
#include <tuple>

class LinApprox16
{
public:
    LinApprox16();
    std::tuple<int, uint16_t, uint16_t> evaluate(const std::vector<uint16_t> &graph);

    struct output_t
    {
        uint16_t index1;
        uint16_t value;
    };

private:
    uint16_t *cuda_graph;
    output_t *cuda_output;
    std::vector<output_t> output;
};
