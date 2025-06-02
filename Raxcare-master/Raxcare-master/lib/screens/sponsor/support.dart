import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SupportGroupScreen extends StatelessWidget {
  const SupportGroupScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Support Groups',
          style: GoogleFonts.poppins(
            fontWeight: FontWeight.w600,
          ),
        ),
        backgroundColor: Colors.orange[700],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: Colors.orange[50],
                  borderRadius: BorderRadius.circular(60),
                ),
                child: Icon(
                  Icons.people_alt,
                  size: 60,
                  color: Colors.orange[700],
                ),
              ),
              const SizedBox(height: 30),
              Text(
                'Support Groups',
                style: GoogleFonts.poppins(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[800],
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                'Coming Soon!',
                style: GoogleFonts.poppins(
                  fontSize: 22,
                  fontWeight: FontWeight.w600,
                  color: Colors.orange[700],
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.2),
                      blurRadius: 15,
                      offset: const Offset(0, 5),
                    ),
                  ],
                  border: Border.all(
                    color: Colors.orange[100]!,
                    width: 1,
                  ),
                ),
                child: Column(
                  children: [
                    Text(
                      'We\'re working hard to bring support groups to this app. Soon you\'ll be able to:',
                      style: GoogleFonts.poppins(
                        fontSize: 16,
                        color: Colors.grey[700],
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 20),
                    _buildFeatureItem(
                      icon: Icons.search,
                      text: 'Find local support groups',
                      color: Colors.blue[700]!,
                    ),
                    const SizedBox(height: 16),
                    _buildFeatureItem(
                      icon: Icons.calendar_month,
                      text: 'View meeting schedules',
                      color: Colors.green[700]!,
                    ),
                    const SizedBox(height: 16),
                    _buildFeatureItem(
                      icon: Icons.star,
                      text: 'Save your favorite groups',
                      color: Colors.amber[700]!,
                    ),
                    const SizedBox(height: 16),
                    _buildFeatureItem(
                      icon: Icons.forum,
                      text: 'Connect with group members',
                      color: Colors.purple[700]!,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 30),
              ElevatedButton(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        'We\'ll notify you when support groups are available!',
                        style: GoogleFonts.poppins(),
                      ),
                      backgroundColor: Colors.orange[700],
                      behavior: SnackBarBehavior.floating,
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange[700],
                  foregroundColor: Colors.white,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  elevation: 4,
                  shadowColor: Colors.orange.withOpacity(0.4),
                ),
                child: Text(
                  'Notify Me When Available',
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureItem({
    required IconData icon,
    required String text,
    required Color color,
  }) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(
            icon,
            color: color,
            size: 24,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Text(
            text,
            style: GoogleFonts.poppins(
              fontSize: 15,
              color: Colors.grey[800],
            ),
          ),
        ),
      ],
    );
  }
}
