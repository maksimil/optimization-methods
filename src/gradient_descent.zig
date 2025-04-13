const config = @import("config.zig");

const Vec2 = config.Vec2;
const Scalar = config.Scalar;

const kArmijoC: Scalar = 1e-4;
const kInitialAlpha: Scalar = 1.0;
const kBacktrackingAlpha: Scalar = 0.5;
const kAlphaThreshold: Scalar = 1e-12;
const kGradTol: Scalar = 1e-8;
const kIterMax: usize = 100_000;

// a + c*b
fn FMulAdd(a: Vec2, b: Vec2, c: Scalar) Vec2 {
    return .{
        a[0] + c * b[0],
        a[1] + c * b[1],
    };
}

fn LineSearch(f: anytype, x0: Vec2, p: Vec2, grad: Vec2) Vec2 {
    const mult = kArmijoC * (p[0] * grad[0] + p[1] * grad[1]);
    var alpha = kInitialAlpha;

    const fx0 = f.call(x0);
    var x1 = Vec2{ 0, 0 };

    while (alpha > kAlphaThreshold) {
        x1 = FMulAdd(x0, p, alpha);
        const fx1 = f.call(x1);

        if (fx1 <= fx0 + alpha * mult) {
            return x1;
        }

        alpha *= kBacktrackingAlpha;
    }

    return x0;
}

pub fn Optimize(f: anytype, grad: anytype, x0: Vec2) Vec2 {
    var x = x0;

    var niter: usize = 0;
    var reached_condition = false;

    while (niter < kIterMax and !reached_condition) {
        const gx = grad.call(x);
        const gx1 = config.Norm1(gx);

        if (gx1 < kGradTol) {
            reached_condition = true;
        } else {
            const x_next = LineSearch(f, x, .{ -gx[0] / gx1, -gx[1] / gx1 }, gx);

            if (x[0] == x_next[0] and x[1] == x_next[1]) {
                reached_condition = true;
            } else {
                x = x_next;
            }
        }

        niter += 1;
    }

    return x;
}
