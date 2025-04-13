const config = @import("config.zig");
const nelder_mead = @import("nelder_mead.zig");
const gradient_descent = @import("gradient_descent.zig");
const conjugate_gradient = @import("conjugate_gradient.zig");

const Scalar = config.Scalar;
const Vec2 = config.Vec2;
const Mat2 = config.Mat2;

const kInitialPoint: Vec2 = .{ 10, 20 };

fn LogResult(
    x: Vec2,
    f: Scalar,
    grad1: Scalar,
    nfcalls: usize,
    ngcalls: usize,
    nhcalls: usize,
) !void {
    try config.stdout.print(
        "x={{ {e:11.4}, {e:11.4} }}, f(x)={e:11.4}, grad1(x)={e:11.4}\n" ++
            "fcalls={d:8}, gcalls={d:8}, hcalls={d:8}\n\n",
        .{ x[0], x[1], f, grad1, nfcalls, ngcalls, nhcalls },
    );
}

pub fn main() !void {
    config.RuntimeInitialize();
    defer config.RuntimeDeinitialize();

    // --- Functions ---
    var nfcalls: usize = 0;
    var ngcalls: usize = 0;
    var nhcalls: usize = 0;

    const f1 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Scalar {
            this.ncalls.* += 1;
            return config.TaskF1(x);
        }
    }{ .ncalls = &nfcalls };

    const f2 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Scalar {
            this.ncalls.* += 1;
            return config.TaskF2(x);
        }
    }{ .ncalls = &nfcalls };

    const g1 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Vec2 {
            this.ncalls.* += 1;
            return config.GradF1(x);
        }
    }{ .ncalls = &ngcalls };

    const g2 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Vec2 {
            this.ncalls.* += 1;
            return config.GradF2(x);
        }
    }{ .ncalls = &ngcalls };

    const h1 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Mat2 {
            this.ncalls.* += 1;
            return config.HessF1(x);
        }
    }{ .ncalls = &nhcalls };

    const h2 = struct {
        ncalls: *usize,

        pub fn call(this: @This(), x: Vec2) Mat2 {
            this.ncalls.* += 1;
            return config.HessF2(x);
        }
    }{ .ncalls = &nhcalls };

    _ = h1;
    _ = h2;

    // --- Nelder-Mead method ---
    try config.stdout.print("\x1B[34mNelder-Mead\x1B[0m\n", .{});
    {
        const res = nelder_mead.Optimize(f1, kInitialPoint);

        try LogResult(
            res,
            config.TaskF1(res),
            config.Norm1(config.GradF1(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }

    {
        const res = nelder_mead.Optimize(f2, kInitialPoint);

        try LogResult(
            res,
            config.TaskF2(res),
            config.Norm1(config.GradF2(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }

    // --- Gradient descent method ---
    try config.stdout.print("\x1B[34mGradient descent\x1B[0m\n", .{});
    {
        const res = gradient_descent.Optimize(f1, g1, kInitialPoint);

        try LogResult(
            res,
            config.TaskF1(res),
            config.Norm1(config.GradF1(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }

    {
        const res = gradient_descent.Optimize(f2, g2, kInitialPoint);

        try LogResult(
            res,
            config.TaskF2(res),
            config.Norm1(config.GradF2(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }

    // --- Conjugate gradient method ---
    try config.stdout.print("\x1B[34mConjugate gradient\x1B[0m\n", .{});
    {
        const res = conjugate_gradient.Optimize(f1, g1, kInitialPoint);

        try LogResult(
            res,
            config.TaskF1(res),
            config.Norm1(config.GradF1(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }

    {
        const res = conjugate_gradient.Optimize(f2, g2, kInitialPoint);

        try LogResult(
            res,
            config.TaskF2(res),
            config.Norm1(config.GradF2(res)),
            nfcalls,
            ngcalls,
            nhcalls,
        );
        nfcalls = 0;
        ngcalls = 0;
        nhcalls = 0;
    }
}
