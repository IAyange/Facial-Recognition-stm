import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'gender_screen.dart';
import '../utils/preferences_service.dart';

class Register extends StatefulWidget {
  final Function toggleView;
  final VoidCallback onRegistrationComplete; // Added callback

  const Register({
    Key? key,
    required this.toggleView,
    required this.onRegistrationComplete,
  }) : super(key: key);

  @override
  _RegisterState createState() => _RegisterState();
}

class _RegisterState extends State<Register> {
  final _formKey = GlobalKey<FormState>();
  bool _loading = false;
  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;

  String _nickname = '';
  String _email = '';
  String _password = '';
  String _confirmPassword = '';
  String _error = '';

  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Future<void> _register() async {
    if (_formKey.currentState!.validate()) {
      setState(() => _loading = true);
      try {
        // Create user with email and password
        final UserCredential result =
            await _auth.createUserWithEmailAndPassword(
          email: _email.trim(),
          password: _password,
        );

        // Store user data in Firestore
        if (result.user != null) {
          await _firestore.collection('users').doc(result.user!.uid).set({
            'nickname': _nickname,
            'email': _email.trim(),
            'createdAt': FieldValue.serverTimestamp(),
            'onboardingComplete': false,
          });

          // Save basic info to preferences
          await PreferencesService.saveData('nickname', _nickname);
          await PreferencesService.saveData('email', _email.trim());
          await PreferencesService.saveData('isNewUser', true as String);

          // Trigger registration complete callback
          widget.onRegistrationComplete();

          // Navigate to personalization screen
          if (mounted) {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => const GenderScreen(),
              ),
            );
          }
        }
      } on FirebaseAuthException catch (e) {
        setState(() {
          _error = e.message ?? 'An error occurred during registration';
          _loading = false;
        });
      } catch (e) {
        setState(() {
          _error = 'An unexpected error occurred';
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 40),
                  Text(
                    'Create Account',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Theme.of(context).colorScheme.primary,
                        ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Join our recovery community',
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          color: Colors.grey[600],
                        ),
                  ),
                  const SizedBox(height: 32),

                  // Nickname Field
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Nickname',
                      hintText: 'How we\'ll address you',
                      prefixIcon: const Icon(Icons.person_outline),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: Colors.grey[50],
                    ),
                    validator: (val) =>
                        val?.isEmpty ?? true ? 'Please enter a nickname' : null,
                    onChanged: (val) => setState(() => _nickname = val),
                  ),
                  const SizedBox(height: 16),

                  // Email Field
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Email',
                      hintText: 'your@email.com',
                      prefixIcon: const Icon(Icons.email_outlined),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: Colors.grey[50],
                    ),
                    validator: (val) {
                      if (val?.isEmpty ?? true) return 'Please enter an email';
                      if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$')
                          .hasMatch(val!)) {
                        return 'Please enter a valid email';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.emailAddress,
                    onChanged: (val) => setState(() => _email = val),
                  ),
                  const SizedBox(height: 16),

                  // Password Field
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Password',
                      hintText: 'At least 6 characters',
                      prefixIcon: const Icon(Icons.lock_outline),
                      suffixIcon: IconButton(
                        icon: Icon(_obscurePassword
                            ? Icons.visibility
                            : Icons.visibility_off),
                        onPressed: () => setState(
                            () => _obscurePassword = !_obscurePassword),
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: Colors.grey[50],
                    ),
                    obscureText: _obscurePassword,
                    validator: (val) {
                      if (val?.isEmpty ?? true)
                        return 'Please enter a password';
                      if (val!.length < 6)
                        return 'At least 6 characters required';
                      return null;
                    },
                    onChanged: (val) => setState(() => _password = val),
                  ),
                  const SizedBox(height: 16),

                  // Confirm Password Field
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Confirm Password',
                      prefixIcon: const Icon(Icons.lock_outline),
                      suffixIcon: IconButton(
                        icon: Icon(_obscureConfirmPassword
                            ? Icons.visibility
                            : Icons.visibility_off),
                        onPressed: () => setState(() =>
                            _obscureConfirmPassword = !_obscureConfirmPassword),
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: Colors.grey[50],
                    ),
                    obscureText: _obscureConfirmPassword,
                    validator: (val) {
                      if (val?.isEmpty ?? true)
                        return 'Please confirm your password';
                      if (val != _password) return 'Passwords do not match';
                      return null;
                    },
                    onChanged: (val) => setState(() => _confirmPassword = val),
                  ),
                  const SizedBox(height: 24),

                  // Error Text
                  if (_error.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Text(
                        _error,
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.error,
                          fontSize: 14,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),

                  // Register Button
                  ElevatedButton(
                    onPressed: _loading ? null : _register,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      backgroundColor: Theme.of(context).colorScheme.primary,
                    ),
                    child: _loading
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Text(
                            'Create Account',
                            style: TextStyle(color: Colors.white),
                          ),
                  ),
                  const SizedBox(height: 16),

                  // Login Link
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Already have an account? ',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                      TextButton(
                        onPressed: () => widget.toggleView(),
                        child: Text(
                          'Sign In',
                          style: TextStyle(
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
