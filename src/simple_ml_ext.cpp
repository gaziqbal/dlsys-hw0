#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

class Matrix
{
private:
public:
    const std::array<size_t, 2> _shape;
    std::shared_ptr<float> _m;

    explicit Matrix(const std::array<size_t, 2> shape, float *m) : _shape(shape), _m(std::shared_ptr<float>(m, [](float *) {}))
    {
    }

    explicit Matrix(const std::array<size_t, 2> shape) : _shape(shape)
    {
        auto dimension = _shape[0] * _shape[1];
        _m = std::shared_ptr<float>(new float[dimension]);
        memset(_m.get(), 0, sizeof(float) * dimension);
    }

    float &at(size_t r, size_t c)
    {
        return _m.get()[(r * _shape[1]) + c];
    }

    const float &at(size_t r, size_t c) const
    {
        return _m.get()[(r * _shape[1]) + c];
    }

    Matrix multiply(const Matrix &other) const
    {
        assert(_shape[1] == other._shape[0]);
        const auto result_shape = std::array<size_t, 2>{_shape[0], other._shape[1]};
        auto result = Matrix(result_shape);
        for (size_t lr = 0; lr < _shape[0]; ++lr)
        {
            for (size_t rc = 0; rc < other._shape[1]; ++rc)
            {
                auto &r = result.at(lr, rc);
                r = 0.0f;
                for (size_t inner = 0; inner < _shape[1]; ++inner)
                {
                    const auto &left = at(lr, inner);
                    const auto &right = other.at(inner, rc);
                    r += left * right;
                }
            }
        }
        return result;
    }

    Matrix subtract(const Matrix &other) const
    {
        assert(_shape == other._shape);
        auto result = Matrix(_shape);
        for (size_t r = 0; r < _shape[0]; ++r)
        {
            for (size_t c = 0; c < _shape[1]; ++c)
            {
                auto &f = result.at(r, c);
                f = at(r, c) - other.at(r, c);
            }
        }
        return result;
    }

    Matrix transpose() const
    {
        auto result = Matrix({_shape[1], _shape[0]});
        for (size_t r = 0; r < _shape[0]; ++r)
        {
            for (size_t c = 0; c < _shape[1]; ++c)
            {
                auto &f = result.at(c, r);
                f = at(r, c);
            }
        }
        return result;
    }

    void print(const std::string &label, bool elements = false) const
    {
        std::cout << "Matrix:" << label << ", Shape: " << _shape[0] << "x" << _shape[1] << std::endl;
        if (elements)
        {
            for (size_t r = 0; r < _shape[0]; ++r)
            {
                std::cout << "[";
                for (size_t c = 0; c < _shape[1]; ++c)
                {
                    std::cout << _m.get()[(r * _shape[1]) + c];
                    std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    void apply(std::function<float(float)> fn)
    {
        const size_t sz = _shape[0] * _shape[1];
        auto p = _m.get();
        for (size_t i = 0; i < sz; ++i)
        {
            *p = fn(*p);
            p++;
        }
    }

    void apply(std::function<float(float, size_t, size_t)> fn)
    {
        auto p = _m.get();
        for (size_t r = 0; r < _shape[0]; ++r)
        {
            for (size_t c = 0; c < _shape[1]; ++c)
            {
                auto e = p + (r * _shape[1]) + c;
                *e = fn(*e, r, c);
            }
        }
    }
};

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    for (size_t i = 0; i < m; i += batch)
    {
        // Batches
        size_t batch_size = std::min(batch, m - i);
        const unsigned char *y_batch = &y[i];
        const float *x_batch = &X[i * n];

        // One hot encoding (batch_size x num_classes)
        Matrix iy({batch_size, k});
        auto one_hot_fn = [&y_batch](float f, size_t r, size_t c)
        {
            const unsigned char label_index = y_batch[r];
            return c == label_index ? 1 : 0;
        };
        iy.apply(one_hot_fn);

        // Source matrices
        Matrix x_batch_matrix({batch_size, n}, const_cast<float *>(x_batch));
        Matrix theta_matrix({n, k}, theta);

        // Softmax probabilities
        Matrix z = x_batch_matrix.multiply(theta_matrix);
        z.apply([](float f)
                { return ::exp(f); });
        auto z_sum = std::make_unique<float[]>(z._shape[0]);
        for (size_t r = 0; r < z._shape[0]; ++r)
        {
            auto sum = 0.0;
            for (size_t c = 0; c < z._shape[1]; ++c)
            {
                sum += z.at(r, c);
            }
            z_sum[r] = sum;
        }
        auto z_norm_fn = [&z_sum](float f, size_t r, size_t c)
        {
            return f / z_sum[r];
        };
        z.apply(z_norm_fn);

        // Gradient
        auto x_batch_transpose_matrix = x_batch_matrix.transpose();
        auto gradient = x_batch_transpose_matrix.multiply(z.subtract(iy));

        // Update theta
        const auto batched_lr = lr / batch;
        for (size_t r = 0; r < n; ++r)
        {
            for (size_t c = 0; c < k; ++c)
            {
                auto &f = theta_matrix.at(r, c);
                f -= batched_lr * gradient.at(r, c);
            }
        }
    }

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
