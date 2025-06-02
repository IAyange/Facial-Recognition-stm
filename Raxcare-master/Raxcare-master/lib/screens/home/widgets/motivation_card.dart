import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:google_generative_ai/google_generative_ai.dart';

class MotivationCard extends StatefulWidget {
  const MotivationCard({Key? key}) : super(key: key);

  @override
  State<MotivationCard> createState() => _MotivationCardState();
}

class _MotivationCardState extends State<MotivationCard> {
  String _quote = "Loading your motivation...";
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchNewQuote();
  }

  Future<void> _fetchNewQuote() async {
    setState(() => _isLoading = true);

    try {
      // Replace with your actual API key
      const apiKey = 'AIzaSyCOutG-g_tVZKzbTtH0bzNjWdoaDVA2YCo';
      final model = GenerativeModel(model: 'gemini-1.5-flash', apiKey: apiKey);

      const prompt =
          "Generate a powerful, emotional quote to help someone resist "
          "relapsing into addiction. Make it raw, real, and impactful. Focus on: "
          "1) The progress they'll lose "
          "3) How much stronger they are than their cravings 4) The brighter future "
          "they're building. Keep it under 2 sentences for mobile screens.";

      final response = await model.generateContent([Content.text(prompt)]);
      setState(() {
        _quote = response.text ?? "Stay strong. You've got this.";
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _quote = "Relapsing steals from the future you deserve. Stay strong.";
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 12,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
      ),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          gradient: const LinearGradient(
            colors: [Color(0xFF6A11CB), Color(0xFF2575FC)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            stops: [0.1, 0.9],
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.deepPurple.withOpacity(0.4),
              blurRadius: 20,
              spreadRadius: 2,
              offset: const Offset(0, 10),
            )
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Stack(
                alignment: Alignment.center,
                children: [
                  const Icon(
                    Icons.favorite,
                    color: Colors.white,
                    size: 40,
                  ),
                  if (_isLoading)
                    const CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                'Your Recovery Armor',
                style: GoogleFonts.poppins(
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 16),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 500),
                child: Text(
                  _quote,
                  key: ValueKey<String>(_quote),
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    color: Colors.white.withOpacity(0.95),
                    fontStyle: FontStyle.italic,
                    height: 1.5,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: _fetchNewQuote,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white.withOpacity(0.2),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  elevation: 0,
                ),
                icon: const Icon(Icons.refresh, size: 20),
                label: Text(
                  'New Motivation',
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
