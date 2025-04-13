const std = @import("std");

pub const Scalar = f64;
pub const kNan = std.math.nan(Scalar);

// runtime
pub var stdout: std.fs.File.Writer = undefined;
pub var stderr: std.fs.File.Writer = undefined;
var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
pub var allocator: std.mem.Allocator = undefined;

pub fn RuntimeInitialize() void {
    stdout = std.io.getStdOut().writer();
    stderr = std.io.getStdErr().writer();

    gpa = @TypeOf(gpa){};
    allocator = gpa.allocator();
}

pub fn RuntimeDeinitialize() void {
    stderr.print("allocator: {}\n", .{gpa.deinit()}) catch unreachable;
}

pub fn ToScalar(x: anytype) Scalar {
    return @as(Scalar, @floatFromInt(x));
}

pub fn Sqr(x: Scalar) Scalar {
    return x * x;
}

// task
pub const Vec2 = [2]Scalar;
pub const Mat2 = [2][2]Scalar;

pub fn Norm1(x: Vec2) Scalar {
    return @abs(x[0]) + @abs(x[1]);
}

pub fn VSqr(x: Vec2) Scalar {
    return x[0] * x[0] + x[1] * x[1];
}

pub fn TaskF1(x: Vec2) Scalar {
    return 100 * Sqr(x[1] - Sqr(x[0])) + 5 * Sqr(1 - x[0]);
}

pub fn GradF1(x: Vec2) Vec2 {
    const c = 100 * 2 * (x[1] - Sqr(x[0]));
    return .{ -2 * c * x[0] - 10 * (1 - x[0]), c };
}

pub fn HessF1(x: Vec2) Mat2 {
    return .{
        .{ -400 * x[1] + 1200 * Sqr(x[0]) + 10, -400 * x[0] },
        .{ -400 * x[0], 200 },
    };
}

pub fn TaskF2(x: Vec2) Scalar {
    return Sqr(Sqr(x[0]) + x[1] - 11) + Sqr(x[0] + Sqr(x[1]) - 7);
}

pub fn GradF2(x: Vec2) Vec2 {
    const c1 = Sqr(x[0]) + x[1] - 11;
    const c2 = x[0] + Sqr(x[1]) - 7;

    return .{ 4 * c1 * x[0] + 2 * c2, 2 * c1 + 4 * c2 * x[1] };
}

pub fn HessF2(x: Vec2) Mat2 {
    return .{
        .{ 12 * Sqr(x[0]) + 4 * Sqr(x[1]) - 42, 4 * x[0] + 4 * x[1] },
        .{ 4 * x[0] + 4 * x[1], 4 * x[0] + 12 * Sqr(x[1]) - 26 },
    };
}
