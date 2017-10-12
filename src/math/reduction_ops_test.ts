/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array1D, Array2D, Scalar} from './ndarray';

// math.min
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);

      expect(math.min(a).get()).toBeCloseTo(-7);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, NaN, 2]);

      expect(math.min(a).get()).toEqual(NaN);

      a.dispose();
    });
  };

  test_util.describeMathCPU('min', [tests]);
  test_util.describeMathGPU('min', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.max
{
  const tests: MathTests = it => {
    it('with one element dominating', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);

      const r = math.max(a);

      expect(r.get()).toBeCloseTo(100);

      a.dispose();
    });

    it('with all elements being the same', math => {
      const a = Array1D.new([3, 3, 3]);

      const r = math.max(a);

      expect(r.get()).toBeCloseTo(3);

      a.dispose();
    });

    it('propagates NaNs', math => {
      expect(math.max(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
    });
  };

  test_util.describeMathCPU('max', [tests]);
  test_util.describeMathGPU('max', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmax
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);
      const result = math.argMax(a);
      expect(result.get()).toBeCloseTo(2);

      a.dispose();
    });

    it('one value', math => {
      const a = Array1D.new([10]);

      const result = math.argMax(a);

      expect(result.get()).toBeCloseTo(0);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([5, 0, 3, NaN, 3]);
      expect(math.argMax(a).get()).toEqual(NaN);
      a.dispose();
    });
  };

  test_util.describeMathCPU('argmax', [tests]);
  test_util.describeMathGPU('argmax', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmin
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);

      const result = math.argMin(a);

      expect(result.get()).toBeCloseTo(1);

      a.dispose();
    });

    it('one value', math => {
      const a = Array1D.new([10]);

      const result = math.argMin(a);

      expect(result.get()).toBeCloseTo(0);

      a.dispose();
    });

    it('Arg min propagates NaNs', math => {
      const a = Array1D.new([5, 0, NaN, 7, 3]);

      expect(math.argMin(a).get()).toEqual(NaN);

      a.dispose();
    });
  };

  test_util.describeMathCPU('argmin', [tests]);
  test_util.describeMathGPU('argmin', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argMaxEquals
{
  const tests: MathTests = it => {
    it('equals', math => {
      const a = Array1D.new([5, 0, 3, 7, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBeCloseTo(1);
    });

    it('not equals', math => {
      const a = Array1D.new([5, 0, 3, 1, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBeCloseTo(0);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([0, 3, 1, 3]);
      const b = Array1D.new([NaN, -20.0, -10.0, -5]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toEqual(NaN);
    });

    it('throws when given arrays of different shape', math => {
      const a = Array1D.new([5, 0, 3, 7, 3, 10]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      expect(() => math.argMaxEquals(a, b)).toThrowError();
    });
  };

  test_util.describeMathCPU('argMaxEquals', [tests]);
  test_util.describeMathGPU('argMaxEquals', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.logSumExp
{
  const tests: MathTests = it => {
    it('0', math => {
      const a = Scalar.new(0);
      const result = math.logSumExp(a);

      test_util.expectNumbersClose(result.get(), 0);

      a.dispose();
      result.dispose();
    });

    it('basic', math => {
      const a = Array1D.new([1, 2, -3]);
      const result = math.logSumExp(a);

      test_util.expectNumbersClose(
          result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));

      a.dispose();
      result.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, 2, NaN]);
      const result = math.logSumExp(a);
      expect(result.get()).toEqual(NaN);
      a.dispose();
    });
  };

  test_util.describeMathCPU('logSumExp', [tests]);
  test_util.describeMathGPU('logSumExp', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sum
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const result = math.sum(a);
      expect(result.get()).toBeCloseTo(7);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      expect(math.sum(a).get()).toEqual(NaN);
      a.dispose();
    });

    it('sum over dtype int32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const sum = math.sum(a);
      expect(sum.get()).toBe(16);
    });

    it('sum over dtype bool', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const sum = math.sum(a);
      expect(sum.get()).toBe(3);
    });
  };

  test_util.describeMathCPU('sum', [tests]);
  test_util.describeMathGPU('sum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
