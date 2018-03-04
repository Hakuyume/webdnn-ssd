use num_traits::Num;
use std::cmp::Ordering;

pub trait Rect<T>
    where T: Copy + PartialOrd + Num
{
    fn x_min(&self) -> T;
    fn y_min(&self) -> T;
    fn x_max(&self) -> T;
    fn y_max(&self) -> T;

    fn area(&self) -> T {
        if self.x_min() < self.x_max() && self.y_min() < self.y_max() {
            (self.x_max() - self.x_min()) * (self.y_max() - self.y_min())
        } else {
            T::zero()
        }
    }

    fn iou<R>(&self, other: &R) -> T
        where R: Rect<T>
    {
        let intersection = || {
            let min = |a: T, b: T| match a.partial_cmp(&b)? {
                Ordering::Greater => Some(b),
                _ => Some(a),
            };

            let max = |a: T, b: T| match a.partial_cmp(&b)? {
                Ordering::Less => Some(b),
                _ => Some(a),
            };

            let x_min = max(self.x_min(), other.x_min())?;
            let y_min = max(self.y_min(), other.y_min())?;
            let x_max = min(self.x_max(), other.x_max())?;
            let y_max = min(self.y_max(), other.y_max())?;

            if x_min < x_max && y_min < y_max {
                Some((x_max - x_min) * (y_max - y_min))
            } else {
                None
            }
        };

        match intersection() {
            Some(i) => i / (self.area() + other.area() - i),
            None => T::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rect;

    struct R(f32, f32, f32, f32);
    impl Rect<f32> for R {
        fn x_min(&self) -> f32 {
            self.0
        }
        fn y_min(&self) -> f32 {
            self.1
        }
        fn x_max(&self) -> f32 {
            self.2
        }
        fn y_max(&self) -> f32 {
            self.3
        }
    }

    #[test]
    fn test_area() {
        assert_eq!(R(0., 2., 6., 9.).area(), 42.);
    }

    #[test]
    fn test_area_empty() {
        assert_eq!(R(0., 2., -4., -2.).area(), 0.);
    }

    #[test]
    fn test_iou() {
        assert_eq!(R(0., 0., 4., 4.).iou(&R(1., 1., 5., 5.)), 9. / 23.);
        assert_eq!(R(0., 0., 4., 4.).iou(&R(4., 0., 8., 4.)), 0.);
    }
}
