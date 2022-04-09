primes = [2]

for number in range(2,100):
    for prime in primes:
        remainder = number % prime
        if remainder is 0:  # The number is not prime
            break
        sqrt = number ** 0.5
        if sqrt < prime:
            primes.append(number)
            break

# m= sqrt(n) ---> m*m = n ,if n is not prime, n = a*b
# n = a*b = m*n
# 1. a>m , b<n
# 2. a=m , b=n
# 3. a<m , b>n


print(primes)